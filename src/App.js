import React, { Component } from 'react';
import * as tf from '@tensorflow/tfjs';

class App extends Component {
  constructor(props) {
    super(props)
    this.state = { hash: undefined, image: undefined, input: undefined, seed: undefined }
    this.generate = this.generate.bind(this)
    this.handleUpload = this.handleUpload.bind(this)
  }

  async generate() {
    // Load the model.
    const model = await tf.loadLayersModel('http://localhost:3000/mnist_model.json');

    // input
    let totalHash = 0
    const splitHash = this.state.hash.split('-')
    splitHash
      .filter(i => i !== '-')
      .forEach(i => {
        const parsedInt = parseInt(i, 16)
        totalHash =+ parsedInt
      })

    const input = tf.randomNormal([1,10], undefined, undefined, undefined, totalHash)
    // const hashedInput = tf.matMul(input, floatHash)
    const inputValues = await input.data()
    // const hashedInputValues = await hashedInput.data()
    // console.log(hashedInputValues)

    this.setState({ input: Array.prototype.slice.call(inputValues), seed: totalHash })

    // Classify the image.
    const generated = await model.predict(input);
    const reshapedGenerated = generated.reshape([28, 28])
    let normalizedGenerated = reshapedGenerated.sub(-1)
    normalizedGenerated = normalizedGenerated.div(2)

    await tf.browser.toPixels(normalizedGenerated, this.refs.canvas);
  }

  handleUpload(e) {
    const files = e.target.files
    const image = URL.createObjectURL(files[0])
    const hash = image.split('blob:http://localhost:3000/')[1]
    this.setState({ hash })
    this.generate()
  }

  render() {
    const { input, seed } = this.state
    return (
      <div>
        <h1>Adversarial Autoencoder</h1>
        <h2>Upload</h2>
        <input type='file' onChange={this.handleUpload} />

        <h2>Seed</h2>
        <p>{seed}</p>

        <h2>Generated Inputs</h2>
        { input
            ? input.map((value, i) => <p key={i}>{`${value.toString()}`}</p>)
            : <div></div>
        }

        <h2>Outputs</h2>
        <canvas ref="canvas" className='canvas' />

        {/*<div>*/}
          {/*<button onClick={this.generate}>Generate</button>*/}
        {/*</div>*/}
      </div>
    );
  }
}

export default App;
