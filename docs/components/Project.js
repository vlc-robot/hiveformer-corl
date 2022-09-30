import React, {Component} from 'react';
import Image from 'next/image';
import BeeHive from '@styles/images/logo.png';
import Model from '@styles/images/model.svg';
import Teaser from '@styles/images/teaser.png';

export default class Project extends Component {
  render() {
    return (
      <>
        <section id="project">
          <div className="row">

            <div className="three columns">

              <Image className="profile-pic" src={BeeHive} alt="Logo of Hiveformer" />

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
            <Image src={Teaser} alt='Teaser figure Hiveformer' />
          </div>
          Hiveformer can adapt to simultaneously perform 74 tasks from <a href='https://arxiv.org/abs/1909.12271' target='_blank' rel='noopener noreferrer'>RLBench</a>  given language
          instructions. Note that tasks can have multiple variations, such as the push buttons task. We test our model on unseen variations on such tasks.
        </div>

        <div className='model'>
          <div className='row'>

            <div className='center'>
              <Image src={Model} alt="Hiveformer's model" />
            </div>
            Hiveformer jointly models instructions, views from multiple cameras, and historical actions and
            observations with a multimodal transformer for robotic manipulation.
          </div>
        </div>

      </>
    );
  }
}
