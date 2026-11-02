// One polyp as a 3D heightfield: the crop lies on the XZ plane, height = elevation above the fitted
// mucosa plane (relative units from polyp3d.py). Uniforms blend in the SAM 2 mask (green), the depth
// map (inferno), dim the mucosa around the polyp and fade it into the background.
import React, {useEffect, useMemo, useState} from 'react';
import * as THREE from 'three';
import {staticFile, delayRender, continueRender, cancelRender} from 'remotion';
import data from '../public/polyp3d/polyps.json';

const G = data.grid;

const vertex = /* glsl */ `
attribute float e;
uniform float uMorph, uH;
varying vec2 vUv;
varying vec3 vPos;
void main() {
  vUv = uv;
  vec3 p = vec3(uv.x * 2.0 - 1.0, e * uH * uMorph, (1.0 - uv.y) * 2.0 - 1.0);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vPos = mv.xyz;
  gl_Position = projectionMatrix * mv;
}`;

const fragment = /* glsl */ `
uniform sampler2D uTex, uDepth, uMask;
uniform float uMaskMix, uDepthMix, uFocus, uShade, uRim;
uniform vec3 uBg;
varying vec2 vUv;
varying vec3 vPos;
void main() {
  float r = length(vUv * 2.0 - 1.0);
  if (r > mix(1.5, 1.0, uFocus)) discard;
  vec3 c = texture2D(uTex, vUv).rgb;
  vec3 col = mix(c, texture2D(uDepth, vUv).rgb, uDepthMix);
  float m = texture2D(uMask, vec2(vUv.x, 1.0 - vUv.y)).r;
  float inside = smoothstep(0.35, 0.65, m);
  col = mix(col, mix(col, vec3(0.18, 0.80, 0.44), 0.45), uMaskMix * inside);
  float rim = clamp((smoothstep(0.2, 0.5, m) - smoothstep(0.5, 0.8, m)) * 2.0, 0.0, 1.0);
  col = mix(col, vec3(0.62, 1.0, 0.76), rim * max(uMaskMix, uRim) * 0.9);
  col *= mix(1.0, mix(0.42, 1.0, inside), uFocus);
  vec3 n = normalize(cross(dFdx(vPos), dFdy(vPos)));
  float sh = 0.58 + 0.42 * max(dot(n, normalize(vec3(0.35, 0.75, 0.55))), 0.0);
  col *= mix(1.0, sh, uShade);
  col = mix(col, uBg, smoothstep(0.72, 1.0, r) * uFocus);
  gl_FragColor = vec4(col, 1.0);
}`;

export type PolypAssets = {elev: Float32Array; mask: THREE.DataTexture; tex: THREE.Texture; depth: THREE.Texture};

// Loads the six polyps (or a subset) once; holds the render until textures and grids are ready.
export const usePolyps = (ids: number[]): PolypAssets[] | null => {
  const [out, setOut] = useState<PolypAssets[] | null>(null);
  const [handle] = useState(() => delayRender(`polyps ${ids.join(',')}`, {timeoutInMilliseconds: 120000}));
  useEffect(() => {
    const loader = new THREE.TextureLoader();
    Promise.all(ids.map(async (n) => {
      const p = data.polyps[n];
      const [eb, mb, tex, depth] = await Promise.all([
        fetch(staticFile(`polyp3d/p${n}.elev.u16`)).then((r) => r.arrayBuffer()),
        fetch(staticFile(`polyp3d/p${n}.mask.u8`)).then((r) => r.arrayBuffer()),
        loader.loadAsync(staticFile(`polyp3d/p${n}.jpg`)),
        loader.loadAsync(staticFile(`polyp3d/p${n}_depth.jpg`)),
      ]);
      const u = new Uint16Array(eb);
      const [e0, e1] = p.elev_range;
      const elev = Float32Array.from(u, (v) => e0 + (v / 65535) * (e1 - e0));
      const mask = new THREE.DataTexture(new Uint8Array(mb), G, G, THREE.RedFormat, THREE.UnsignedByteType);
      mask.minFilter = THREE.LinearFilter;
      mask.magFilter = THREE.LinearFilter;
      mask.needsUpdate = true;
      tex.anisotropy = 8;
      return {elev, mask, tex, depth};
    })).then((a) => { setOut(a); continueRender(handle); }).catch((e) => cancelRender(e));
  }, [handle]);
  return out;
};

type Props = {
  a: PolypAssets; morph?: number; height?: number; maskMix?: number; depthMix?: number;
  focus?: number; shade?: number; rim?: number; spin?: number; bg?: string;
};

export const PolypMesh: React.FC<Props> = ({
  a, morph = 1, height = 1.6, maskMix = 0, depthMix = 0, focus = 1, shade = 1, rim = 0, spin = 0, bg = '#0B0F14',
}) => {
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const uv = new Float32Array(G * G * 2);
    for (let j = 0; j < G; j++) {
      for (let i = 0; i < G; i++) {
        const k = j * G + i;
        uv[2 * k] = i / (G - 1);
        uv[2 * k + 1] = 1 - j / (G - 1);
      }
    }
    const idx: number[] = [];
    for (let j = 0; j < G - 1; j++) {
      for (let i = 0; i < G - 1; i++) {
        const p = j * G + i;
        idx.push(p, p + G, p + 1, p + 1, p + G, p + G + 1);
      }
    }
    g.setIndex(idx);
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(G * G * 3), 3));
    g.setAttribute('e', new THREE.BufferAttribute(a.elev, 1));
    return g;
  }, [a]);

  const mat = useMemo(() => new THREE.ShaderMaterial({
    vertexShader: vertex, fragmentShader: fragment, side: THREE.DoubleSide,
    uniforms: {
      uTex: {value: a.tex}, uDepth: {value: a.depth}, uMask: {value: a.mask}, uMorph: {value: 0}, uH: {value: 1},
      uMaskMix: {value: 0}, uDepthMix: {value: 0}, uFocus: {value: 1}, uShade: {value: 1}, uRim: {value: 0},
      uBg: {value: new THREE.Color(bg)},
    },
  }), [a, bg]);
  const u = mat.uniforms;
  u.uMorph.value = morph;
  u.uH.value = height;
  u.uMaskMix.value = maskMix;
  u.uDepthMix.value = depthMix;
  u.uFocus.value = focus;
  u.uShade.value = shade;
  u.uRim.value = rim;

  return <mesh geometry={geom} material={mat} rotation={[0, spin, 0]} frustumCulled={false} />;
};
