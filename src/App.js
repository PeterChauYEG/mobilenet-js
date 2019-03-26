import React, { Component } from 'react';
import * as mobilenet from '@tensorflow-models/mobilenet';

class App extends Component {
  constructor(props) {
    super(props)
    this.state = { image: undefined, predictions: [] }
    this.classify = this.classify.bind(this)
    this.onUpload = this.onUpload.bind(this)
  }

  async classify() {
    // Load the model.
    const model = await mobilenet.load();

    // Classify the image.
    const predictions = await model.classify(this.refs.image);

    this.setState({ predictions })
  }

  onUpload(e) {
    const files = e.target.files
    const image = URL.createObjectURL(files[0])
    console.log(files[0])
    this.setState({ image })
  }

  render() {
    return (
      <div>
        <input type='file' id='upload' onChange={this.onUpload} />
        <img src={this.state.image} ref='image' className='image' />

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
