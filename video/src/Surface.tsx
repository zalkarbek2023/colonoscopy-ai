// A colonoscopy frame as a 3D surface: every depth-grid vertex is back-projected through a pinhole
// camera in the vertex shader, z = 1 / (A + B * d), d = relative inverse depth (1 = near).
// uMorph blends from a flat picture (z = const) to the lifted surface; uDepthMix blends the photo
// into an inferno depth map; the detected polyp box is drawn as a green ellipse on the surface.
import React, {useMemo} from 'react';
import * as THREE from 'three';
import {useThree} from '@react-three/fiber';

export const TAN_X = Math.tan((100 / 2) * (Math.PI / 180)); // assumed 100° horizontal field of view
export const Z_FLAT = 1.4;

const vertex = /* glsl */ `
attribute float d;
uniform float uMorph, uTan, uAspect, uA, uB, uZf;
varying vec2 vUv;
varying float vD;
varying vec3 vPos;
void main() {
  vUv = uv;
  vD = d;
  float z = mix(uZf, 1.0 / (uA + uB * d), uMorph);
  vec3 p = vec3((uv.x * 2.0 - 1.0) * uTan * z, (uv.y * 2.0 - 1.0) * uTan / uAspect * z, -z);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vPos = mv.xyz;
  gl_Position = projectionMatrix * mv;
}`;

const fragment = /* glsl */ `
uniform sampler2D uTex;
uniform float uDepthMix, uHi, uShade, uWire, uOpacity;
uniform vec4 uBox;
varying vec2 vUv;
varying float vD;
varying vec3 vPos;
vec3 inferno(float t) {
  t = clamp(t, 0.0, 1.0);
  const vec3 c0 = vec3(0.0002189, 0.0016510, -0.0194809);
  const vec3 c1 = vec3(0.1065134, 0.5639564, 3.9327124);
  const vec3 c2 = vec3(11.6024931, -3.9728540, -15.9423941);
  const vec3 c3 = vec3(-41.7039961, 17.4363989, 44.3541452);
  const vec3 c4 = vec3(77.1629357, -33.4023589, -81.8073093);
  const vec3 c5 = vec3(-71.3194282, 32.6260643, 73.2095199);
  const vec3 c6 = vec3(25.1311262, -12.2426690, -23.0703250);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}
void main() {
  vec3 c = texture2D(uTex, vUv).rgb;
  if (dot(c, vec3(0.299, 0.587, 0.114)) < 0.035) discard; // black corners outside the endoscope's view
  if (uWire > 0.5) { gl_FragColor = vec4(1.0, 0.54, 0.30, uOpacity); return; }
  vec3 col = mix(c, inferno(vD), uDepthMix);
  vec2 q = vec2(vUv.x, 1.0 - vUv.y);
  vec2 cen = (uBox.xy + uBox.zw) * 0.5;
  vec2 rad = max((uBox.zw - uBox.xy) * 0.5, vec2(1e-4));
  float e = length((q - cen) / rad);
  float inside = (1.0 - smoothstep(0.9, 1.0, e)) * uHi;
  float rim = (smoothstep(0.82, 0.93, e) - smoothstep(0.98, 1.06, e)) * uHi;
  col = mix(col, vec3(0.18, 0.80, 0.44), inside * 0.32);
  col = mix(col, vec3(0.62, 1.0, 0.76), rim * 0.9);
  vec3 n = normalize(cross(dFdx(vPos), dFdy(vPos)));
  float sh = 0.70 + 0.30 * abs(dot(n, normalize(vec3(0.35, 0.55, 1.0))));
  col *= mix(1.0, sh, uShade);
  gl_FragColor = vec4(col, uOpacity);
}`;

type Props = {
  gw: number; gh: number; depth: Uint16Array; offset?: number; tex: THREE.Texture;
  morph: number; depthMix?: number; highlight?: number; box?: number[] | null;
  shade?: number; step?: number; wire?: boolean; opacity?: number; a?: number; b?: number;
};

export const Surface: React.FC<Props> = ({
  gw, gh, depth, offset = 0, tex, morph, depthMix = 0, highlight = 0, box = null,
  shade = 1, step = 1, wire = false, opacity = 1, a = 0.35, b = 1.0,
}) => {
  const w = Math.floor((gw - 1) / step) + 1;
  const h = Math.floor((gh - 1) / step) + 1;
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const uv = new Float32Array(w * h * 2);
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) {
        const k = j * w + i;
        uv[2 * k] = (i * step) / (gw - 1);
        uv[2 * k + 1] = 1 - (j * step) / (gh - 1);
      }
    }
    const idx: number[] = [];
    for (let j = 0; j < h - 1; j++) {
      for (let i = 0; i < w - 1; i++) {
        const p = j * w + i;
        idx.push(p, p + w, p + 1, p + 1, p + w, p + w + 1);
      }
    }
    g.setIndex(idx);
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(w * h * 3), 3)); // computed in the shader
    g.setAttribute('d', new THREE.BufferAttribute(new Float32Array(w * h), 1));
    return g;
  }, [gw, gh, w, h, step]);

  useMemo(() => {
    const attr = geom.getAttribute('d') as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) arr[j * w + i] = depth[offset + j * step * gw + i * step] / 65535;
    }
    attr.needsUpdate = true;
  }, [geom, depth, offset, w, h, step, gw]);

  const mat = useMemo(() => new THREE.ShaderMaterial({
    vertexShader: vertex, fragmentShader: fragment, side: THREE.DoubleSide, wireframe: wire,
    transparent: wire || opacity < 1, depthWrite: !wire,
    uniforms: {
      uTex: {value: tex}, uMorph: {value: 0}, uTan: {value: TAN_X}, uAspect: {value: gw / gh},
      uA: {value: a}, uB: {value: b}, uZf: {value: Z_FLAT}, uDepthMix: {value: 0}, uHi: {value: 0},
      uShade: {value: 1}, uWire: {value: wire ? 1 : 0}, uOpacity: {value: 1}, uBox: {value: new THREE.Vector4()},
    },
  }), [wire, gw, gh]);

  const u = mat.uniforms;
  u.uTex.value = tex;
  u.uMorph.value = morph;
  u.uDepthMix.value = depthMix;
  u.uShade.value = shade;
  u.uOpacity.value = opacity;
  u.uA.value = a;
  u.uB.value = b;
  u.uHi.value = box ? highlight : 0;
  if (box) (u.uBox.value as THREE.Vector4).set(box[0], box[1], box[2], box[3]);
  mat.transparent = wire || opacity < 1;

  return <mesh geometry={geom} material={mat} frustumCulled={false} />;
};

// Places the camera every frame; target is the point it looks at.
export const Cam: React.FC<{pos: [number, number, number]; target: [number, number, number]; fov: number}> = ({pos, target, fov}) => {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera;
  camera.position.set(...pos);
  camera.fov = fov;
  camera.near = 0.01;
  camera.far = 100;
  camera.updateProjectionMatrix();
  camera.lookAt(...target);
  return null;
};

// Camera on a sphere around a target: azimuth/elevation in degrees, radius in scene units.
export const orbit = (target: [number, number, number], az: number, el: number, r: number): [number, number, number] => {
  const A = (az * Math.PI) / 180;
  const E = (el * Math.PI) / 180;
  return [target[0] + r * Math.sin(A) * Math.cos(E), target[1] + r * Math.sin(E), target[2] + r * Math.cos(A) * Math.cos(E)];
};
