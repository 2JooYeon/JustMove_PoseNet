import React, { useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as posenet from '@tensorflow-models/posenet';
import cosineSimilarity from 'cosine-similarity';
import { drawKeypoints, drawSkeleton, drawBoundingBox } from '../utilities';
import original from '../img/flank.jpg';
import target from '../img/target.jpg';

const Similarity = () => {
  const similarity = require('compute-cosine-similarity');
  const canvasRef = useRef(null);
  const originImgRef = useRef(null);
  const targetImgRef = useRef(null);

  /* 유사도 측정하기 */
  const runPosenet = async () => {
    const net = await posenet.load({
      architecture: 'ResNet50',
      outputStride: 32,
      inputResolution: { width: 257, height: 200 },
      quantBytes: 2,
    });

    // 원본 이미지 바운딩박스랑 키포인트 추적
    const originPose = await keyDetect(net, originImgRef);
    const originKeyPoints = originPose['keypoints'];
    const originBoundingBox = posenet.getBoundingBoxPoints(originKeyPoints);

    // 도전 이미지 바운딩박스랑 키포인트 추적
    const targetPose = await keyDetect(net, targetImgRef);
    const targetKeyPoints = targetPose['keypoints'];
    const targetBoundingBox = posenet.getBoundingBoxPoints(targetKeyPoints);


    // 원본 키포인트 고차원 벡터로 변환
    const compressedOriginKeyPoints = compressKeyVectors(originKeyPoints);

    // 도전 키포인트 고차원 벡터로 변환
    const compressedTargetKeypoints = compressKeyVectors(targetKeyPoints);

 
    // 원본 이미지 바운딩박스랑 키포인트 원점기준으로 평행이동
    adjustPoints(compressedOriginKeyPoints, originBoundingBox);

    // 도전 이미지 바운딩박스랑 키포인트 원점기준으로 평행이동
    adjustPoints(compressedTargetKeypoints, targetBoundingBox);


    // 원본 키포인트와 도전 키포인트 단위벡터로 정규화
    const originUnitVectors = convertToUnitVector(compressedOriginKeyPoints);
    const targetUnitVectors = convertToUnitVector(compressedTargetKeypoints);

    // 코사인 유사도 측정
    const cosineSimilarity = computeCosineSimilarty(originUnitVectors, targetUnitVectors);





  };



  /* PoseNet을 이용한 포즈 추정 결과 받아오기 (신뢰도 점수 및 키포인트 좌표) */
  const keyDetect = async (net, img, pose) => {
    if (typeof img.current !== 'undefined' && img.current !== null) {

      const image = img.current;

      // Make Detections
      const pose = await net.estimateSinglePose(image, {
        flipHorizontal : true
      });

      return pose;
    }
  };


  /* PoseNet 결과로 나온 키포인트 좌표 고차원 벡터로 압축하기 */
  const compressKeyVectors = (keypoints) => {
    let compressedVectors = [];

    for (let pose of keypoints) {
      compressedVectors.push(pose['position']['x']);
      compressedVectors.push(pose['position']['y']);
    }
    return compressedVectors;
  };



  /* BoundingBox와 KeyPoints를 원점으로 이동시키는 함수 */
  const adjustPoints = (keypoints, boundingBox) => {

    // BoundingBox와 KeyPoints를 원점으로 이동시키기 위한 양
    const movement_x = boundingBox[0]['x'];
    const movement_y = boundingBox[0]['y'];

    // BoundingBox 원점으로 이동시키기
    for (let point of boundingBox) {
      point['x'] -= movement_x;
      point['y'] -= movement_y;
    }

    // KeyPoint 원점으로 이동시키기
    for (let pose of keypoints) {
      pose['position']['x'] -= movement_x;
      pose['position']['y'] -= movement_y;
    }
  };



  /* 고차원 벡터를 단위벡터로 변환하기*/
  const convertToUnitVector= (compressedKeypoints) => {
    const unitVectors = compressedKeypoints;
    let sum =0;

    for(let keypoint of compressedKeypoints){
        sum += Math.pow(keypoint, 2);
    }

    for(let keypoint of compressKeyVectors){
        unitVectors /= Math.sqrt(sum);
    }

    return unitVectors;
  }



  /* 코사인 유사도 측정하기 */
  const computeCosineSimilarty= (unitOriginKeyPoints, unitTargetKeyPoints) => {
    const cosineSimilarity = similarity(unitOriginKeyPoints, unitTargetKeyPoints);
    return cosineSimilarity;
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
