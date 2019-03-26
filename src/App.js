import React, { Component } from 'react';
import * as mobilenet from '@tensorflow-models/mobilenet';
import cat from './cat.png';

class App extends Component {
  constructor(props) {
    super(props)
    this.state = { predictions: [] }
    this.classify = this.classify.bind(this)
  }

  async classify() {
    // Load the model.
    const model = await mobilenet.load();

    // Classify the image.
    const predictions = await model.classify(this.refs.cat);

    this.setState({ predictions })
  }

  render() {
    return (
      <div>
        <img src={cat} ref='cat' className='image' />

        {this.state.predictions.map((prediction, i) => {
          const { className, probability } = prediction
          return (<p key={i}>{`Prediction: ${className} (${probability*100}%)`}</p>)
        })}

        <button onClick={this.classify}>Classify</button>
      </div>
    );
  }
}

export default App;
