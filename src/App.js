import React, { Component } from 'react';
import * as tf from '@tensorflow/tfjs';

class App extends Component {
  constructor(props) {
    super(props)
    this.state = { image: undefined, input: undefined }
    this.generate = this.generate.bind(this)
    this.handleUpload = this.handleUpload.bind(this)
  }

  async generate() {
    // Load the model.
    const model = await tf.loadLayersModel('http://localhost:3000/mnist_model.json');

    // input
    const input = tf.browser.fromPixels(this.refs.image).asType('float32')
    const resizedInput = tf.image.resizeNearestNeighbor(input, [28, 28])
    console.log(resizedInput)

    // Classify the image.
    const generated = await model.predict(resizedInput);
    const reshapedGenerated = generated.reshape([28, 28])
    let normalizedGenerated = reshapedGenerated.sub(-1)
    normalizedGenerated = normalizedGenerated.div(2)

    await tf.browser.toPixels(normalizedGenerated, this.refs.canvas);
  }

  handleUpload(e) {
    const files = e.target.files
    const image = URL.createObjectURL(files[0])
    this.setState({ image })
  }

  render() {
    const { image, input } = this.state
    return (
      <div>
        <h1>Adversarial Autoencoder</h1>
        <h2>Upload</h2>
        <input type='file' onChange={this.handleUpload} />
        <img src={image} ref='image' className='image' />

        <h2>Generated Inputs</h2>
        { input
            ? input.map((value, i) => <p key={i}>{`${value.toString()}`}</p>)
            : <div></div>
        }

        <h2>Outputs</h2>
        <canvas ref="canvas" className='canvas' />

        <div>
          <button onClick={this.generate}>Generate</button>
        </div>
      </div>
    );
  }
}

export default App;
