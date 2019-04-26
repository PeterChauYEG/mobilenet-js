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
    const generator = await tf.loadLayersModel('http://localhost:3000/generator/model.json');
    // const discriminator = await tf.loadLayersModel('http://localhost:3000/discriminator/model.json');

    // input
    const input = tf.browser.fromPixels(this.refs.image).asType('float32')
    const resizedInput = tf.image.resizeNearestNeighbor(input, [1, 10]).div(255)
    const grayScaledInput = tf.mean(resizedInput, 2, )

    await tf.browser.toPixels(grayScaledInput, this.refs.inputCanvas);


    // Classify the image.
    // const encoded = await discriminator.predict(resizedInput);
    const decoded = await generator.predict(grayScaledInput);
    const reshapedGenerated = decoded.reshape([28, 28])
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

        <h2>Preprocessed Inputs</h2>
        <canvas ref="inputCanvas" className='preprocessedCanvas' />

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
