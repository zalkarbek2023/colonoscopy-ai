// "Colonoscopy AI, now in 3D": one REAL-Colon frame → polyp detection → single-frame depth → 3D surface.
// Style follows the lecture trailer (paper grid, warm edge, one statement per beat). All numbers come from
// public/colon3d/scenes.json, written by medai_colon/depth3d.py + export_web.py on the RTX 5060 Ti.
import React, {useEffect, useState} from 'react';
import {
  AbsoluteFill, Sequence, Img, staticFile, useCurrentFrame, interpolate, Easing,
  delayRender, continueRender, cancelRender,
} from 'remotion';
import {ThreeCanvas} from '@remotion/three';
import * as THREE from 'three';
import {loadFont as loadOnest} from '@remotion/google-fonts/Onest';
import {loadFont as loadMono} from '@remotion/google-fonts/JetBrainsMono';
import scenes from '../public/colon3d/scenes.json';
import {Surface, Cam, orbit} from './Surface';

export const {fontFamily: SANS} = loadOnest('normal', {weights: ['400', '600', '800'], subsets: ['latin']});
export const {fontFamily: MONO} = loadMono('normal', {weights: ['400', '500'], subsets: ['latin']});

export const INK = '#0E141B';
export const PAPER = '#EFEEEA';
export const SUB = '#3D4651';
export const ORANGE = '#FF8A4C';
export const PINK = '#FF3D6E';
export const GREEN = '#2ECC71';
export const VIEW_BG = '#0B0F14';

export const CLAMP = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;
export const OUT = Easing.bezier(0.16, 1, 0.3, 1);
export const IN_OUT = Easing.bezier(0.65, 0, 0.35, 1);
export const ramp = (f: number, a: number, b: number, e = OUT) => interpolate(f, [a, b], [0, 1], {...CLAMP, easing: e});
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

const S = {hook: 0, lift: 120, tunnel: 720, live: 900, code: 1110, outro: 1290};
export const TOTAL = 1440;

const [GW, GH] = scenes.grid;
const VIEW = {x: 760, y: 120, w: 1060, h: 840};
const TARGET: [number, number, number] = [0, 0, -1.6];

const gradText: React.CSSProperties = {
  backgroundImage: `linear-gradient(90deg, ${ORANGE}, ${PINK})`,
  WebkitBackgroundClip: 'text', backgroundClip: 'text', WebkitTextFillColor: 'transparent',
};

// ---- text primitives (same motion as the lecture trailer)
export const Words: React.FC<{
  text: string; at?: number; size?: number; color?: string; weight?: number;
  gradientLine?: number; style?: React.CSSProperties;
}> = ({text, at = 0, size = 96, color = INK, weight = 800, gradientLine = -1, style}) => {
  const f = useCurrentFrame();
  let k = 0;
  return (
    <div style={{position: 'absolute', fontFamily: SANS, fontSize: size, fontWeight: weight, lineHeight: 1.06, letterSpacing: -size * 0.03, color, ...style}}>
      {text.split('|').map((line, li) => (
        <div key={li} style={{display: 'flex', flexWrap: 'wrap', columnGap: size * 0.24}}>
          {line.split(' ').map((w, wi) => {
            const p = ramp(f, at + k * 3, at + k * 3 + 16);
            k++;
            return (
              <span key={wi} style={{
                display: 'inline-block', opacity: p,
                transform: `translateY(${(1 - p) * size * 0.4}px)`, filter: `blur(${(1 - p) * 10}px)`,
                ...(li === gradientLine ? gradText : {}),
              }}>{w}</span>
            );
          })}
        </div>
      ))}
    </div>
  );
};

export const Mono: React.FC<{children: React.ReactNode; at?: number; size?: number; color?: string; style?: React.CSSProperties}> = ({
  children, at = 0, size = 26, color = SUB, style,
}) => {
  const f = useCurrentFrame();
  const p = ramp(f, at, at + 14);
  return (
    <div style={{position: 'absolute', fontFamily: MONO, fontSize: size, lineHeight: 1.5, letterSpacing: 0.3, color, opacity: p, transform: `translateY(${(1 - p) * 12}px)`, ...style}}>
      {children}
    </div>
  );
};

// Visible only between frames a and b (local), with short fades.
export const Window: React.FC<{a: number; b: number; children: React.ReactNode}> = ({a, b, children}) => {
  const f = useCurrentFrame();
  const o = Math.min(ramp(f, a, a + 10, IN_OUT), 1 - ramp(f, b - 10, b, IN_OUT));
  if (f < a || f > b) return null;
  return <Sequence from={a} layout="none"><AbsoluteFill style={{opacity: o}}>{children}</AbsoluteFill></Sequence>;
};

export const Scene: React.FC<{dur: number; fadeOut?: boolean; children: React.ReactNode}> = ({dur, fadeOut = true, children}) => {
  const f = useCurrentFrame();
  const o = Math.min(ramp(f, 0, 8, IN_OUT), fadeOut ? 1 - ramp(f, dur - 8, dur, IN_OUT) : 1);
  return <AbsoluteFill style={{opacity: o}}>{children}</AbsoluteFill>;
};

export const Paper: React.FC = () => {
  const f = useCurrentFrame();
  const s = Math.sin((f / 30) * 0.8);
  return (
    <AbsoluteFill style={{background: PAPER}}>
      <AbsoluteFill style={{
        backgroundImage: 'linear-gradient(rgba(14,20,27,0.055) 1px, transparent 1px), linear-gradient(90deg, rgba(14,20,27,0.055) 1px, transparent 1px)',
        backgroundSize: '48px 48px',
      }} />
      <AbsoluteFill style={{
        background: `linear-gradient(90deg, rgba(239,238,234,0) ${72 + s * 2}%, rgba(255,61,110,0.26) ${89 + s}%, rgba(255,138,76,0.9) 100%)`,
      }} />
    </AbsoluteFill>
  );
};

const Chrome: React.FC = () => {
  const f = useCurrentFrame();
  const starts = Object.values(S);
  const idx = starts.filter((s) => f >= s).length - 1;
  return (
    <>
      <div style={{position: 'absolute', left: 116, top: 48, padding: '6px 12px', borderRadius: 6, background: 'rgba(239,238,234,0.92)', fontFamily: MONO, fontSize: 22, letterSpacing: 4, color: SUB}}>
        COLONOSCOPY AI · 3D
      </div>
      <svg style={{position: 'absolute', left: 1582, top: 64}} width={216} height={8}>
        {starts.map((_, i) => (
          <rect key={i} x={i * 37} width={32} height={8} rx={4} fill={INK} fillOpacity={i < idx ? 0.4 : i === idx ? 1 : 0.12} />
        ))}
      </svg>
    </>
  );
};

// ---- data: depth grids (uint16) and textures, loaded once per scene
type Assets = {depth: Uint16Array; tex: THREE.Texture[]};
const useAssets = (name: 'polyp' | 'tunnel' | 'seq'): Assets | null => {
  const [data, setData] = useState<Assets | null>(null);
  const [handle] = useState(() => delayRender(`assets ${name}`, {timeoutInMilliseconds: 120000}));
  useEffect(() => {
    const loader = new THREE.TextureLoader();
    const urls = name === 'seq'
      ? Array.from({length: scenes.seq.n}, (_, k) => staticFile(`colon3d/seq/${String(k).padStart(3, '0')}.jpg`))
      : [staticFile(`colon3d/${name}.jpg`)];
    Promise.all([
      fetch(staticFile(`colon3d/${name}.u16`)).then((r) => r.arrayBuffer()),
      Promise.all(urls.map((u) => loader.loadAsync(u))),
    ]).then(([buf, tex]) => {
      tex.forEach((t) => { t.anisotropy = 8; t.generateMipmaps = true; });
      setData({depth: new Uint16Array(buf), tex});
      continueRender(handle);
    }).catch((e) => cancelRender(e));
  }, [handle, name]);
  return data;
};

// Dark rounded window holding the 3D canvas.
const Viewport: React.FC<{children: React.ReactNode; enter?: number}> = ({children, enter = 1}) => (
  <div style={{
    position: 'absolute', left: VIEW.x, top: VIEW.y + (1 - enter) * 40, width: VIEW.w, height: VIEW.h, borderRadius: 18,
    overflow: 'hidden', background: VIEW_BG, opacity: enter, boxShadow: '0 18px 54px rgba(14,20,27,0.28)',
  }}>
    <ThreeCanvas width={VIEW.w} height={VIEW.h} gl={{antialias: true, preserveDrawingBuffer: true}}>
      <color attach="background" args={[VIEW_BG]} />
      {children}
    </ThreeCanvas>
  </div>
);

const Step: React.FC<{n: string; title: string; lines: React.ReactNode}> = ({n, title, lines}) => (
  <>
    <Mono at={0} size={24} color={ORANGE} style={{left: 124, top: 300, letterSpacing: 4}}>STEP {n}</Mono>
    <Words text={title} at={4} size={84} style={{left: 118, top: 350, width: 600}} />
    <Mono at={24} style={{left: 124, top: 580, width: 600}}>{lines}</Mono>
  </>
);

// ---- scene 0: the hook
const Hook: React.FC = () => (
  <AbsoluteFill>
    <Words text="A colonoscope|sees in 2D." at={6} size={150} style={{left: 128, top: 250}} />
    <Words text="The colon is 3D." at={44} size={150} gradientLine={0} style={{left: 128, top: 590}} />
  </AbsoluteFill>
);

// ---- scene 1: detect → depth → lift, one continuous 3D shot of the polyp frame
const LiftScene: React.FC = () => {
  const f = useCurrentFrame();
  const a = useAssets('polyp');
  const box = scenes.polyp.box;
  const hi = Math.max(ramp(f, 24, 54) * (1 - ramp(f, 150, 172)), ramp(f, 470, 510));
  const depthMix = ramp(f, 160, 220, IN_OUT) * (1 - ramp(f, 420, 480, IN_OUT));
  const morph = ramp(f, 300, 420, IN_OUT);
  const swing = ramp(f, 300, 430, IN_OUT);
  const back = ramp(f, 440, 600, IN_OUT);
  const az = lerp(0, 42, swing) + lerp(0, -70, back);
  const el = lerp(0, 16, swing) + lerp(0, -4, back);
  const r = lerp(5.4, 7.6, swing);
  const wire = ramp(f, 300, 330) * (1 - ramp(f, 420, 450));
  // face-on label above the box while the picture is still flat
  const sx = (x: number) => VIEW.x + VIEW.w / 2 + (x * 2 - 1) * 0.949 * (VIEW.w / 2);
  const sy = (y: number) => VIEW.y + VIEW.h / 2 + (y * 2 - 1) * 0.958 * (VIEW.h / 2);
  return (
    <AbsoluteFill>
      {a && (
        <Viewport enter={ramp(f, 0, 20)}>
          <Cam pos={orbit(TARGET, az, el, r)} target={TARGET} fov={30} />
          <Surface gw={GW} gh={GH} depth={a.depth} tex={a.tex[0]} morph={morph} depthMix={depthMix}
            highlight={hi} box={box} shade={morph} b={0.9} />
          {wire > 0 && <Surface gw={GW} gh={GH} depth={a.depth} tex={a.tex[0]} morph={morph} step={4} wire opacity={0.55 * wire} b={0.9} />}
        </Viewport>
      )}
      {box && (
        <Window a={30} b={150}>
          <div style={{
            position: 'absolute', left: sx(box[0]), top: sy(box[1]) - 52, padding: '6px 12px', borderRadius: 6,
            background: GREEN, color: '#05140B', fontFamily: MONO, fontSize: 24, fontWeight: 500,
          }}>polyp · {scenes.polyp.conf.toFixed(2)}</div>
        </Window>
      )}
      <Window a={8} b={156}>
        <Step n="01" title="Find|the polyp." lines={<>YOLO26m · TensorRT FP16<br />{scenes.det_ms} ms per frame</>} />
      </Window>
      <Window a={156} b={306}>
        <Step n="02" title="Depth from|one frame." lines={<>Depth Anything V2 Small<br />{scenes.depth_ms} ms · bright = near</>} />
      </Window>
      <Window a={306} b={600}>
        <Step n="03" title="Lift every pixel|into 3D." lines={<>{(GW * GH).toLocaleString('en-US')} vertices<br />pinhole back-projection<br /><span style={{color: GREEN}}>the polyp, in 3D</span></>} />
      </Window>
      <Mono at={20} size={20} style={{left: VIEW.x, top: VIEW.y + VIEW.h + 22, width: VIEW.w}}>
        REAL-Colon · video {scenes.video} · t = {scenes.polyp.t.toFixed(1)} s · never seen in training
      </Mono>
    </AbsoluteFill>
  );
};

// ---- scene 2: the open lumen, flying into the colon
const TunnelScene: React.FC = () => {
  const f = useCurrentFrame();
  const a = useAssets('tunnel');
  const morph = ramp(f, 6, 50, IN_OUT);
  const fly = ramp(f, 40, 175, IN_OUT);
  const pos: [number, number, number] = [lerp(0, 0.35, fly), lerp(0, -0.2, fly), lerp(3.6, 0.5, fly)];
  return (
    <AbsoluteFill>
      {a && (
        <Viewport>
          <Cam pos={pos} target={[0, 0, -3.2]} fov={lerp(30, 58, fly)} />
          <Surface gw={GW} gh={GH} depth={a.depth} tex={a.tex[0]} morph={morph} shade={morph} />
        </Viewport>
      )}
      <Words text="Same math,|open lumen." at={6} size={84} style={{left: 118, top: 350, width: 600}} />
      <Mono at={30} style={{left: 124, top: 580, width: 600}}>dark centre = far<br />folds around the scope = near</Mono>
      <Mono at={20} size={20} style={{left: VIEW.x, top: VIEW.y + VIEW.h + 22, width: VIEW.w}}>
        REAL-Colon · video {scenes.video} · t = {scenes.tunnel.t.toFixed(1)} s
      </Mono>
    </AbsoluteFill>
  );
};

// ---- scene 3: the clip, frame by frame in 3D, with the 2D frame as an inset
const LiveScene: React.FC = () => {
  const f = useCurrentFrame();
  const a = useAssets('seq');
  const n = scenes.seq.n;
  const k = Math.min(n - 1, Math.floor((f / 30) * scenes.seq.fps));
  const box = scenes.seq.box[k];
  const az = lerp(34, 22, f / 210);
  const insetW = 360;
  const insetH = (insetW * GH) / GW;
  return (
    <AbsoluteFill>
      {a && (
        <Viewport>
          <Cam pos={orbit(TARGET, az, 14, 7.2)} target={TARGET} fov={30} />
          <Surface gw={GW} gh={GH} depth={a.depth} offset={k * GW * GH} tex={a.tex[k]} morph={1} highlight={1} box={box} b={0.9} />
        </Viewport>
      )}
      <Words text="Frame|by frame." at={6} size={84} style={{left: 118, top: 250, width: 600}} />
      <Mono at={26} style={{left: 124, top: 450, width: 600}}>
        detection {scenes.det_ms} ms + depth {scenes.depth_ms} ms<br />≈ {Math.round(1000 / (scenes.det_ms + scenes.depth_ms))} FPS model time · one RTX 5060 Ti
      </Mono>
      <div style={{position: 'absolute', left: 124, top: 640, width: insetW, height: insetH, borderRadius: 10, overflow: 'hidden', background: '#000', opacity: ramp(f, 16, 30)}}>
        <Img src={staticFile(`colon3d/seq/${String(k).padStart(3, '0')}.jpg`)} style={{width: insetW, height: insetH}} />
        {box && (
          <div style={{
            position: 'absolute', left: box[0] * insetW, top: box[1] * insetH, width: (box[2] - box[0]) * insetW,
            height: (box[3] - box[1]) * insetH, border: `3px solid ${GREEN}`, borderRadius: 4,
          }} />
        )}
      </div>
      <Mono at={30} size={20} style={{left: 124, top: 640 + insetH + 14, width: insetW + 200}}>2D input · t = {scenes.seq.t[k].toFixed(1)} s</Mono>
    </AbsoluteFill>
  );
};

// ---- scene 4: the code that does it
const CODE: [string, string][] = [
  ['#', '1 · polyp: YOLO26m, TensorRT FP16'],
  ['py', 'r = det.predict(frame, imgsz=960, conf=0.25)[0]'],
  ['', ''],
  ['#', '2 · depth: Depth Anything V2 Small, relative'],
  ['py', 'disp = depth_net(pixel_values=px).predicted_depth'],
  ['', ''],
  ['#', '3 · lift every pixel into 3D (vertex shader)'],
  ['gl', 'float z = 1.0 / (A + B * d);     // near → small z'],
  ['gl', 'vec3 p  = vec3((u*2.-1.) * tanX * z,'],
  ['gl', '               (v*2.-1.) * tanX / aspect * z, -z);'],
];

const CodeScene: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill>
      <Words text="The whole idea|in three steps." at={6} size={84} style={{left: 118, top: 300, width: 600}} />
      <Mono at={34} style={{left: 124, top: 530, width: 560}}>medai_colon/depth3d.py<br />video/src/Surface.tsx</Mono>
      <div style={{
        position: 'absolute', left: VIEW.x, top: VIEW.y + 80, width: VIEW.w, height: 680, borderRadius: 18, background: VIEW_BG,
        boxShadow: '0 18px 54px rgba(14,20,27,0.28)', padding: '54px 56px', boxSizing: 'border-box', opacity: ramp(f, 0, 16),
      }}>
        {CODE.map(([kind, line], i) => {
          const p = ramp(f, 14 + i * 9, 26 + i * 9);
          const shown = line.slice(0, Math.round(line.length * Math.min(1, p * 1.4)));
          return (
            <div key={i} style={{
              fontFamily: MONO, fontSize: 27, lineHeight: '50px', whiteSpace: 'pre', opacity: p,
              color: kind === '#' ? '#7C8794' : kind === 'gl' ? '#FFB38A' : '#E6EDF3',
            }}>{kind === '#' ? `# ${shown}` : shown || ' '}</div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

// ---- scene 5: credits and the link
const Outro: React.FC = () => (
  <AbsoluteFill>
    <Words text="Colonoscopy AI,|now in 3D." at={4} size={130} gradientLine={1} style={{left: 128, top: 210}} />
    <Mono at={30} size={34} color={INK} style={{left: 134, top: 540, width: 1500}}>github.com/zalkarbek2023/colonoscopy-ai · open source</Mono>
    <Mono at={44} size={22} style={{left: 134, top: 640, width: 1500}}>
      Research demo, not a medical device. Depth is relative, not millimetres.<br />
      Data: REAL-Colon (Biffi et al., Sci Data 2024), CC BY 4.0 · Depth Anything V2 Small, Apache-2.0
    </Mono>
    <Mono at={60} size={26} color={INK} style={{left: 134, top: 800, width: 1500}}>Zalkarbek Arapbaev · MD & AI engineer</Mono>
  </AbsoluteFill>
);

export const Colon3D: React.FC = () => (
  <AbsoluteFill style={{fontFamily: SANS}}>
    <Paper />
    <Sequence from={S.hook} durationInFrames={S.lift - S.hook}><Scene dur={S.lift - S.hook}><Hook /></Scene></Sequence>
    <Sequence from={S.lift} durationInFrames={S.tunnel - S.lift}><Scene dur={S.tunnel - S.lift}><LiftScene /></Scene></Sequence>
    <Sequence from={S.tunnel} durationInFrames={S.live - S.tunnel}><Scene dur={S.live - S.tunnel}><TunnelScene /></Scene></Sequence>
    <Sequence from={S.live} durationInFrames={S.code - S.live}><Scene dur={S.code - S.live}><LiveScene /></Scene></Sequence>
    <Sequence from={S.code} durationInFrames={S.outro - S.code}><Scene dur={S.outro - S.code}><CodeScene /></Scene></Sequence>
    <Sequence from={S.outro} durationInFrames={TOTAL - S.outro}><Scene dur={TOTAL - S.outro} fadeOut={false}><Outro /></Scene></Sequence>
    <Chrome />
  </AbsoluteFill>
);
