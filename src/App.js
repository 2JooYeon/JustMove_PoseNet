// Steps to apply posenet to react webcam

// 1. Install dependencies OK
// 2. Import dependencies OK
// 3. Setup webcam and canvas OK
// 4. Define references to those OK
// 5. Load posenet OK
// 6. Detect function OK
// 7. Drawing utilities from tensorflow OK
// 8. Draw functions OK

import React, { useRef } from 'react';
import './App.css';
import * as tf from '@tensorflow/tfjs';
import * as posenet from '@tensorflow-models/posenet';
import Webcam from 'react-webcam';
import { drawKeypoints, drawSkeleton } from './utilities';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  // Load posenet
  const runPosenet = async () => {
    const net = await posenet.load({
      architecture: 'ResNet50',
      outputStride: 32,
      inputResolution: { width: 257, height: 200 },
      quantBytes: 2,
    });

    setInterval(() => {
      detect(net);
    }, 100);
  };

  const detect = async (net) => {
    if (
      typeof webcamRef.current !== 'undefined' &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = 640;
      const videoHeight = 480;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Make Detections
      const pose = await net.estimateSinglePose(video, {
        flipHorizontal: true,
      });
      console.log(pose);

      // drawCanvas(pose, video, videoWidth, videoHeight, canvasRef);
      requestAnimationFrame(() => {
        drawCanvas(pose, video, videoWidth, videoHeight, canvasRef);
      });
    }
  };
  const drawCanvas = (pose, video, videoWidth, videoHeight, canvas) => {
    const ctx = canvas.current.getContext('2d');
    canvas.current.width = videoWidth;
    canvas.current.height = videoHeight;

    drawKeypoints(pose['keypoints'], 0.6, ctx);
    drawSkeleton(pose['keypoints'], 0.7, ctx);
  };

  runPosenet();

  return (
    <div className="App">
      <header className="App-header">
        <Webcam
          mirrored={true}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          style={{
            position: 'absolute',
            marginLeft: 'auto',
            marginRight: 'auto',
            left: 0,
            right: 0,
            textAlign: 'center',
            zindex: 9,
            width: 640,
            height: 480,
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            marginLeft: 'auto',
            marginRight: 'auto',
            left: 0,
            right: 0,
            textAlign: 'center',
            zindex: 9,
            width: 640,
            height: 480,
          }}
        />
      </header>
    </div>
  );
}

export default App;
