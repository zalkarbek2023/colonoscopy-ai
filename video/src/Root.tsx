import React from 'react';
import {Composition} from 'remotion';
import {Colon3D, TOTAL} from './Colon3D';
import {Polyps3D, TOTAL as POLYPS_TOTAL} from './Polyps3D';

export const Root: React.FC = () => (
  <>
    <Composition id="Polyps3D" component={Polyps3D} durationInFrames={POLYPS_TOTAL} fps={30} width={1920} height={1080} />
    <Composition id="Colon3D" component={Colon3D} durationInFrames={TOTAL} fps={30} width={1920} height={1080} />
  </>
);
