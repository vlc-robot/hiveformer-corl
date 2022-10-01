import React, {Component} from 'react';
import Logo from '@styles/images/logo.png';

export default class About extends Component {
  render() {
    return (
      <section id="about">
        <div className="row">

          <div className="three columns">

            <Image src={Logo} className="profile-pic" alt="A bee hive" />

          </div>

          <div className="nine columns main-col">

            <h2>About Me</h2>
            <p>
            </p>

            <div className="row">

              <div className="columns contact-details">

                <h2>Contact Details</h2>
                <p className="address">
                  <span></span>
                  <br></br>
                  <span>

                  </span>
                  <br></br>
                  <span></span>
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    );
  }
}
