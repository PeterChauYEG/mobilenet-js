import React, { Component } from 'react';
import * as tf from '@tensorflow/tfjs';

class App extends Component {
  constructor(props) {
    super(props)
    this.state = { image: undefined, input: undefined }
    this.generate = this.generate.bind(this)
  }

  async generate() {
    // Load the model.
    const model = await tf.loadLayersModel('http://localhost:3000/mnist_model.json');

    // input
    const input = tf.randomNormal([1,10])
    const inputValues = await input.data()
    this.setState({ input: Array.prototype.slice.call(inputValues) })

    // Classify the image.
    const generated = await model.predict(input);
    const reshapedGenerated = generated.reshape([28, 28])
    let normalizedGenerated = reshapedGenerated.sub(-1)
    normalizedGenerated = normalizedGenerated.div(2)

    await tf.browser.toPixels(normalizedGenerated, this.refs.canvas);
  }

  render() {
    const { input } = this.state
    return (
      <div>
        <h1>Adversarial Autoencoder</h1>
        <h2>Inputs</h2>
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
