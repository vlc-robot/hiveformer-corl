import React, {Component} from 'react';

export default class Project extends Component {
  render() {
    return (
      <>
        <section id="project">
          <div className="row">

            <div className="three columns">

              <img className="profile-pic" src="images/logo.png" alt="Logo of Hiveformer" />

            </div>

            <div className="nine columns main-col">

              <h1>Instruction-driven history-aware policies
                for robotic manipulations</h1>
              <p>
                Pierre-Louis Guhur<sup>1</sup>,
                Shizhe Chen <sup>1</sup>,
                Ricardo Garcia <sup>1</sup>,
                Makarand Tapaswi <sup>2</sup>,
                Ivan Laptev <sup>1</sup>,
                Cordelia Schmid <sup>1</sup>
              </p>

              <p><sup>1</sup>Inria, École normale supérieure, CNRS, PSL Research University <br />
                <sup>2</sup>IIIT Hyderabad</p>

            </div>
          </div>

        </section>
        <div className='row teaser'>
          <div className='center'>
            <img src='images/teaser.png' alt='Teaser figure Hiveformer' />
          </div>
          Hiveformer can adapt to simultaneously perform 74 tasks from <a href='https://arxiv.org/abs/1909.12271' target='_blank' rel='noopener noreferrer'>RLBench</a>  given language
          instructions. Note that tasks can have multiple variations, such as the push buttons task. We test our model on unseen variations on such tasks.
        </div>

        <div className='model'>
          <div className='row'>

            <div className='center'>
              <img src='images/model.svg' alt="Hiveformer's model" />
            </div>
            Hiveformer jointly models instructions, views from multiple cameras, and historical actions and
            observations with a multimodal transformer for robotic manipulation.
          </div>
        </div>

      </>
    );
  }
}
