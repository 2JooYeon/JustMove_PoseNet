import React, { useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as posenet from '@tensorflow-models/posenet';
import { poseSimilarity } from 'posenet-similarity';
import { drawKeypoints, drawSkeleton, drawBoundingBox } from '../utilities';
import original from '../img/flank.jpg';
import target from '../img/target.jpg';

const Similarity = () => {
  const canvasRef = useRef(null);
  const originImgRef = useRef(null);
  const targetImgRef = useRef(null);

  // PoseNet 돌리기 
    const runPosenet = async () => {
    const net = await posenet.load({
      architecture: 'ResNet50',
      outputStride: 32,
      inputResolution: { width: 257, height: 200 },
      quantBytes: 2,
    });

    keyDetect(net);
  };



  const keyDetect = async (net) => {
    if (
      typeof originImgRef.current !== 'undefined' &&
      originImgRef.current !== null &&
      typeof targetImgRef.current !== 'undefined' &&
      targetImgRef.current !== null
    ) {
      // Get original image properties
      const originImage = originImgRef.current;
      const originWidth = originImgRef.current.width;
      const originHeight = originImgRef.current.height;

      // Get target image properties
      const targetImage = targetImgRef.current;
      const targetWidth = targetImgRef.current.width;
      const targetHeight = targetImgRef.current.height;

      // Make Detections
      const originPose = await net.estimateSinglePose(originImage);
      const targetPose = await net.estimateSinglePose(targetImage);

      console.log('원본 이미지 좌표' + originPose);
      console.log('타겟 이미지 좌표' + targetPose);

      const originBoundingBox = posenet.getBoundingBoxPoints(
        originPose['keypoints']
      );

      const targetBoundingBox = posenet.getBoundingBoxPoints(
        targetPose['keypoints']
      );

      console.log(originBoundingBox);
      console.log(targetBoundingBox);

      drawCanvas(originPose, originImage, originWidth, originHeight, canvasRef);
    }
  };

  const compressKeyVectors= (keypoints) => {
    const oneDimKeyVectors = [];

    for(let pose in keypoints){
      oneDimKeyVectors.push(pose['position']['x']);
      oneDimKeyVectors.push(pose['position']['y']);
    }
    console.log(oneDimKeyVectors);
    
  }

  const drawCanvas = (pose, video, videoWidth, videoHeight, canvas) => {
    const ctx = canvas.current.getContext('2d');
    canvas.current.width = videoWidth;
    canvas.current.height = videoHeight;

    drawBoundingBox(pose['keypoints'], ctx);
    drawKeypoints(pose['keypoints'], 0.6, ctx);
    drawSkeleton(pose['keypoints'], 0.8, ctx);
  };

  runPosenet();

  return (
    <div className="App">
      <header className="App-header">
        <img ref={originImgRef} src={original} />
        <img ref={targetImgRef} src={target} />

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
          }}
        />
      </header>
    </div>
  );
};

export default Similarity;
