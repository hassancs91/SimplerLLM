import React from 'react';
import { useCurrentFrame, useVideoConfig, interpolate, Easing, AbsoluteFill } from 'remotion';

export const compositionConfig = {
  id: 'MinecraftScene',
  durationInSeconds: 5,
  fps: 30,
  width: 1080,
  height: 1920,
};

const blockTexture = 'url(/path/to/pixelated-texture.png)';
const treeTexture = 'url(/path/to/tree-texture.png)';
const leafTexture = 'url(/path/to/leaf-texture.png)';
const orbTexture = 'url(/path/to/green-diamond-orb-texture.png)';

const orbs = Array.from({length: 5}, (_, i) => ({
  id: i,
  start: i * 10,
}));

const MinecraftScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  const hillHeight = interpolate(frame, [0, fps * 2], [0, 200], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.cubic),
  });

  const treeGrow = interpolate(frame, [fps * 2, fps * 3], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.cubic),
  });

  const leafPop = interpolate(frame, [fps * 3, fps * 4], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.cubic),
  });

  const orbAnimations = orbs.map(orb => {
    const rotation = interpolate(
      frame - orb.start,
      [0, fps],
      [0, 360],
      {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      }
    );
    const bounce = interpolate(
      frame - orb.start,
      [0, fps / 2, fps],
      [0, -20, 0],
      {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      }
    );
    return { rotation, bounce };
  });

  return (
    <AbsoluteFill style={{backgroundColor: '#87CEEB'}}>
      <div
        style={{
          position: 'absolute',
          bottom: '15%',
          left: '50%',
          transform: 'translate(-50%, 0)',
          width: '200px',
          height: `${hillHeight}px`,
          backgroundImage: blockTexture,
        }}
      />

      <div
        style={{
          position: 'absolute',
          bottom: `${15 + hillHeight}px`,
          left: '50%',
          transform: 'translate(-50%, 0)',
          width: '50px',
          height: '100px',
          backgroundImage: treeTexture,
          opacity: treeGrow,
        }}
      />

      <div
        style={{
          position: 'absolute',
          bottom: `${15 + hillHeight + 100}px`,
          left: '50%',
          transform: 'translate(-50%, 0)',
          width: '50px',
          height: '50px',
          backgroundImage: leafTexture,
          opacity: leafPop,
        }}
      />

      {orbAnimations.map((anim, i) => (
        <div
          key={i}
          style={{
            position: 'absolute',
            bottom: `${15 + hillHeight + anim.bounce}px`,
            left: `${10 + (i * 15)}%`,
            width: '16px',
            height: '16px',
            backgroundImage: orbTexture,
            transform: `rotate(${anim.rotation}deg)`,
          }}
        />
      ))}
    </AbsoluteFill>
  );
};

export default MinecraftScene;
