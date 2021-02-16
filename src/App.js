// Steps to apply posenet to react webcam
// 1. Install dependencies OK
// 2. Import dependencies OK
// 3. Setup webcam and canvas OK
// 4. Define references to those
// 5. Load posenet
// 6. Detect function
// 7. Drawing utilities from tensorflow
// 8. Draw functions

import React, { useRef } from 'react';
import './App.css';
import * as tf from '@tensorflow/tfjs';
import * as posenet from '@tensorflow-models/posenet';
import Webcam from 'react-webcam';

function App() {
  return (
    <div className="App">
      <header className="App-header">Hello world!</header>
    </div>
  );
}

export default App;
