// "Polyps in 3D": our detector finds the polyp, SAM 2 cuts it out, single-frame depth lifts it off the
// wall; then an atlas of six real polyps with their expert size and histology (REAL-Colon labels).
// Data: public/polyp3d, written by medai_colon/polyp_views.py + polyp3d.py.
import React from 'react';
import {AbsoluteFill, Sequence, Img, staticFile, useCurrentFrame} from 'remotion';
import {ThreeCanvas} from '@remotion/three';
import data from '../public/polyp3d/polyps.json';
import {Cam, orbit} from './Surface';
import {PolypMesh, usePolyps} from './Polyp';
import {
  Words, Mono, Window, Scene, Paper, ramp, lerp, IN_OUT, SANS, MONO, INK, SUB, ORANGE, GREEN, VIEW_BG,
} from './Colon3D';

const S = {hook: 0, steps: 120, atlas: 660, code: 1020, outro: 1200};
export const TOTAL = 1350;

const VIEW = {x: 760, y: 120, w: 1060, h: 840};
const SQUARE = 740; // on-screen size of the crop in the top-down 3D view (fov 35°, r 3.6)

const NAMES = ['adenoma', 'traditional serrated adenoma', 'adenoma', 'sessile serrated lesion', 'adenoma, LST-NG', 'hyperplastic polyp'];
const SITE: Record<string, string> = {
  transverse: 'transverse colon', rectum: 'rectum', ascending: 'ascending colon', descending: 'descending colon', sigma: 'sigmoid colon',
};
const P = data.polyps;
const hero = P[0];

const Chrome: React.FC = () => {
  const f = useCurrentFrame();
  const starts = Object.values(S);
  const idx = starts.filter((s) => f >= s).length - 1;
  return (
    <>
      <div style={{position: 'absolute', left: 116, top: 48, padding: '6px 12px', borderRadius: 6, background: 'rgba(239,238,234,0.92)', fontFamily: MONO, fontSize: 22, letterSpacing: 4, color: SUB}}>
        POLYPS · 3D
      </div>
      <svg style={{position: 'absolute', left: 1619, top: 64}} width={180} height={8}>
        {starts.map((_, i) => (
          <rect key={i} x={i * 37} width={32} height={8} rx={4} fill={INK} fillOpacity={i < idx ? 0.4 : i === idx ? 1 : 0.12} />
        ))}
      </svg>
    </>
  );
};

const Step: React.FC<{n: string; title: string; lines: React.ReactNode}> = ({n, title, lines}) => (
  <>
    <Mono at={0} size={24} color={ORANGE} style={{left: 124, top: 280, letterSpacing: 4}}>STEP {n}</Mono>
    <Words text={title} at={4} size={84} style={{left: 118, top: 330, width: 600}} />
    <Mono at={24} style={{left: 124, top: 560, width: 600}}>{lines}</Mono>
  </>
);

const Hook: React.FC = () => (
  <AbsoluteFill>
    <Words text="A polyp's shape|matters." at={6} size={150} gradientLine={1} style={{left: 128, top: 250}} />
    <Mono at={50} size={30} style={{left: 134, top: 640, width: 1500}}>
      pedunculated · sessile · flat: the Paris classification is about shape.<br />An endoscope shows it in 2D.
    </Mono>
  </AbsoluteFill>
);

// Full frame with our detector's box, zooming into the square crop that the 3D view starts from.
const FindView: React.FC<{t: number; boxIn: number}> = ({t, boxIn}) => {
  const [fw, fh] = data.frame_size;
  const [x0, y0, side] = hero.crop;
  const s0 = Math.min(VIEW.w / fw, VIEW.h / fh);
  const s1 = SQUARE / side;
  const s = s0 * Math.pow(s1 / s0, t);
  const fx = lerp(fw / 2, x0 + side / 2, t);
  const fy = lerp(fh / 2, y0 + side / 2, t);
  const left = VIEW.w / 2 - fx * s;
  const top = VIEW.h / 2 - fy * s;
  const b = hero.found?.box;
  return (
    <div style={{position: 'absolute', left: VIEW.x, top: VIEW.y, width: VIEW.w, height: VIEW.h, borderRadius: 18, overflow: 'hidden', background: VIEW_BG, boxShadow: '0 18px 54px rgba(14,20,27,0.28)'}}>
      <Img src={staticFile('polyp3d/hero_frame.jpg')} style={{position: 'absolute', left, top, width: fw * s, height: fh * s}} />
      {b && (
        <>
          <div style={{
            position: 'absolute', left: left + b[0] * s, top: top + b[1] * s, width: (b[2] - b[0]) * s, height: (b[3] - b[1]) * s,
            border: `4px solid ${GREEN}`, borderRadius: 6, opacity: boxIn * (1 - t), transform: `scale(${lerp(1.15, 1, boxIn)})`,
          }} />
          <div style={{
            position: 'absolute', left: left + b[0] * s, top: top + b[1] * s - 48, padding: '6px 12px', borderRadius: 6,
            background: GREEN, color: '#05140B', fontFamily: MONO, fontSize: 24, fontWeight: 500, opacity: boxIn * (1 - t),
          }}>polyp · {hero.found?.conf.toFixed(2)}</div>
        </>
      )}
    </div>
  );
};

const StepsScene: React.FC = () => {
  const f = useCurrentFrame();
  const a = usePolyps([0]);
  const zoom = ramp(f, 100, 148, IN_OUT);
  const show3d = ramp(f, 146, 156);
  const maskMix = ramp(f, 160, 200) * (1 - ramp(f, 270, 300));
  const depthMix = ramp(f, 275, 320, IN_OUT) * (1 - ramp(f, 410, 460, IN_OUT));
  const morph = ramp(f, 390, 480, IN_OUT);
  const tilt = ramp(f, 390, 480, IN_OUT);
  const focus = ramp(f, 420, 470);
  const rim = ramp(f, 470, 500);
  const el = lerp(89.5, 30, tilt);
  const az = lerp(0, 35, tilt) + lerp(0, 45, ramp(f, 480, 540, (x) => x));
  const target: [number, number, number] = [0, lerp(0, 0.15, tilt), 0];
  return (
    <AbsoluteFill>
      {f < 158 && <div style={{opacity: ramp(f, 0, 20)}}><FindView t={zoom} boxIn={ramp(f, 22, 42)} /></div>}
      {a && f >= 140 && (
        <div style={{position: 'absolute', left: VIEW.x, top: VIEW.y, width: VIEW.w, height: VIEW.h, borderRadius: 18, overflow: 'hidden', background: VIEW_BG, opacity: show3d}}>
          <ThreeCanvas width={VIEW.w} height={VIEW.h} gl={{antialias: true, preserveDrawingBuffer: true}}>
            <color attach="background" args={[VIEW_BG]} />
            <Cam pos={orbit(target, az, el, lerp(3.6, 3.3, tilt))} target={target} fov={35} />
            <PolypMesh a={a[0]} morph={morph} height={1.8} maskMix={maskMix} depthMix={depthMix} focus={focus} shade={morph} rim={rim} />
          </ThreeCanvas>
        </div>
      )}
      <Window a={8} b={150}>
        <Step n="01" title="Find|the polyp." lines={<>our YOLO26m detector<br />on a video it never saw in training</>} />
      </Window>
      <Window a={150} b={270}>
        <Step n="02" title="Cut it out." lines={<>SAM 2, prompted with the box<br /><span style={{color: GREEN}}>pixel mask of the polyp</span></>} />
      </Window>
      <Window a={270} b={390}>
        <Step n="03" title="Estimate|depth." lines={<>Depth Anything V2 Small<br />one frame · bright = near</>} />
      </Window>
      <Window a={390} b={540}>
        <Step n="04" title="Lift it|off the wall." lines={<>fit a plane to the mucosa around it<br />keep what rises above</>} />
      </Window>
      <Window a={470} b={540}>
        <div style={{position: 'absolute', left: 124, top: 700, padding: '16px 20px', borderRadius: 12, background: INK, color: '#fff', width: 600}}>
          <div style={{fontFamily: SANS, fontSize: 36, fontWeight: 800}}>{hero.size_mm} mm · {NAMES[0]}</div>
          <div style={{fontFamily: MONO, fontSize: 20, color: '#AEB6C0', marginTop: 6}}>{SITE[hero.site]} · expert labels, REAL-Colon</div>
        </div>
      </Window>
      <Mono at={20} size={20} style={{left: VIEW.x, top: VIEW.y + VIEW.h + 22, width: VIEW.w}}>
        REAL-Colon · video {hero.video} · t = {hero.t} s
      </Mono>
    </AbsoluteFill>
  );
};

const CELL = {w: 520, h: 330};
const AtlasScene: React.FC = () => {
  const f = useCurrentFrame();
  const a = usePolyps([0, 1, 2, 3, 4, 5]);
  return (
    <AbsoluteFill>
      <Words text="Six polyps, six shapes." at={4} size={64} style={{left: 118, top: 110}} />
      {a && P.map((p, n) => {
        const x = 120 + (n % 3) * (CELL.w + 30);
        const y = 220 + Math.floor(n / 3) * (CELL.h + 80);
        const p0 = ramp(f, 10 + n * 6, 30 + n * 6);
        const morph = ramp(f, 30 + n * 6, 90 + n * 6, IN_OUT);
        return (
          <div key={n} style={{position: 'absolute', left: x, top: y + (1 - p0) * 30, opacity: p0}}>
            <div style={{width: CELL.w, height: CELL.h, borderRadius: 14, overflow: 'hidden', background: VIEW_BG}}>
              <ThreeCanvas width={CELL.w} height={CELL.h} gl={{antialias: true, preserveDrawingBuffer: true}}>
                <color attach="background" args={[VIEW_BG]} />
                <Cam pos={orbit([0, 0.15, 0], 20 + n * 25 + f * 0.45, 30, 2.8)} target={[0, 0.15, 0]} fov={35} />
                <PolypMesh a={a[n]} morph={morph} height={1.5} shade={1} focus={1} rim={0.6} />
              </ThreeCanvas>
            </div>
            <div style={{fontFamily: SANS, fontSize: 28, fontWeight: 800, color: INK, marginTop: 10}}>{p.size_mm} mm · {NAMES[n]}</div>
            <div style={{fontFamily: MONO, fontSize: 17, color: SUB, marginTop: 2, width: CELL.w}}>
              {SITE[p.site]} · {p.video}{n === 0 ? ' · found by our model' : ''}
            </div>
          </div>
        );
      })}
      <Mono at={60} size={18} style={{left: 1000, top: 122, width: 800}}>
        Size and histology: expert labels from REAL-Colon.<br />3D shape: single-frame depth, relative, not a measurement.
      </Mono>
    </AbsoluteFill>
  );
};

const CODE: [string, string][] = [
  ['#', '1 · find: our YOLO26m'],
  ['py', 'r = det.predict(frame, imgsz=960, conf=0.25)[0]'],
  ['#', '2 · cut out: SAM 2, prompted with the box'],
  ['py', 'mask = sam(crop, bboxes=[box])[0].masks.data[0]'],
  ['#', '3 · depth: Depth Anything V2 Small, relative'],
  ['py', 'disp = depth_net(pixel_values=px).predicted_depth'],
  ['#', '4 · lift: what rises above the wall'],
  ['py', 'plane = fit_plane(disp, ring)       # mucosa around it'],
  ['py', 'elev = (disp - plane) / (hi - lo)   # same units for all'],
];

const CodeScene: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill>
      <Words text="Four steps,|open source." at={6} size={84} style={{left: 118, top: 320, width: 600}} />
      <Mono at={34} style={{left: 124, top: 550, width: 560}}>medai_colon/polyp3d.py<br />video/src/Polyp.tsx</Mono>
      <div style={{
        position: 'absolute', left: VIEW.x, top: VIEW.y + 60, width: VIEW.w, height: 720, borderRadius: 18, background: VIEW_BG,
        boxShadow: '0 18px 54px rgba(14,20,27,0.28)', padding: '54px 56px', boxSizing: 'border-box', opacity: ramp(f, 0, 16),
      }}>
        {CODE.map(([kind, line], i) => {
          const p = ramp(f, 14 + i * 9, 26 + i * 9);
          const shown = line.slice(0, Math.round(line.length * Math.min(1, p * 1.4)));
          return (
            <div key={i} style={{
              fontFamily: MONO, fontSize: 27, lineHeight: '56px', whiteSpace: 'pre', opacity: p,
              color: kind === '#' ? '#7C8794' : '#E6EDF3', marginTop: kind === '#' && i > 0 ? 14 : 0,
            }}>{kind === '#' ? `# ${shown}` : shown}</div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

const Outro: React.FC = () => (
  <AbsoluteFill>
    <Words text="Polyps,|in 3D." at={4} size={150} gradientLine={1} style={{left: 128, top: 170}} />
    <Mono at={30} size={34} color={INK} style={{left: 134, top: 540, width: 1500}}>github.com/zalkarbek2023/colonoscopy-ai · open source</Mono>
    <Mono at={44} size={22} style={{left: 134, top: 640, width: 1600}}>
      Research demo, not a medical device. Shapes come from single-frame depth: relative, not millimetres.<br />
      Data and expert labels: REAL-Colon (Biffi et al., Sci Data 2024), CC BY 4.0 · SAM 2 and Depth Anything V2 Small, Apache-2.0
    </Mono>
    <Mono at={60} size={26} color={INK} style={{left: 134, top: 800, width: 1500}}>Zalkarbek Arapbaev · MD & AI engineer</Mono>
  </AbsoluteFill>
);

export const Polyps3D: React.FC = () => (
  <AbsoluteFill style={{fontFamily: SANS}}>
    <Paper />
    <Sequence from={S.hook} durationInFrames={S.steps - S.hook}><Scene dur={S.steps - S.hook}><Hook /></Scene></Sequence>
    <Sequence from={S.steps} durationInFrames={S.atlas - S.steps}><Scene dur={S.atlas - S.steps}><StepsScene /></Scene></Sequence>
    <Sequence from={S.atlas} durationInFrames={S.code - S.atlas}><Scene dur={S.code - S.atlas}><AtlasScene /></Scene></Sequence>
    <Sequence from={S.code} durationInFrames={S.outro - S.code}><Scene dur={S.outro - S.code}><CodeScene /></Scene></Sequence>
    <Sequence from={S.outro} durationInFrames={TOTAL - S.outro}><Scene dur={TOTAL - S.outro} fadeOut={false}><Outro /></Scene></Sequence>
    <Chrome />
  </AbsoluteFill>
);
