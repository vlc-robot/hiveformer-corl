import React, {Component} from 'react';
import * as constants from '@services/constants';


export default class Header extends Component {
  render() {
    return (
      <React.Fragment>

        <header id="home">
          <nav id="nav-wrap">
            <a className="mobile-btn" href="#nav-wrap" title="Show navigation">Show navigation</a>
            <a className="mobile-btn" href="#hide" title="Hide navigation">Hide navigation</a>
            { /* <ul id="nav" className="nav">
              <li className="current"><a className="smoothscroll" href="#home">Home</a></li>
              <li><a className="smoothscroll" href="#about">Demo</a></li>
              <li><a className="smoothscroll" href="#resume">Paper</a></li>
              {
                constants.socialLinks && Object.entries(constants.socialLinks).map(([name, item]) => {
                  if (item.disable) return null;
                  return (
                    <li key={item.name}>
                      <a href={item.url} className='smoothscroll' target="_blank" rel='noopener noreferrer' >
                        <i className={`fa ${item.icon}`}></i> {name}
                      </a>
                    </li>
                  )
                }
                )
              }
            </ul>
            */}
          </nav>

          <div className="row banner">
            <div className="banner-text">
              <h1 className="responsive-headline">Hiveformer</h1>
              <h3 style={{color: '#fff', fontFamily: 'sans-serif '}}>
                A new model to solve robotics tasks using instructions, multi-views and history.
              </h3>
              <hr />
              <ul className="social">
                {
                  constants.codeLinks && Object.entries(constants.codeLinks).map(([name, item], index) => {
                    if (item.disable) return null;
                    return (
                      <li key={index}>
                        <a href={item.url} target="_blank" rel='noopener noreferrer' ><i className={`fa ${item.icon}`}></i> {name}</a>
                      </li>
                    )
                  }
                  )
                }
              </ul>
              <ul className="social">
                {
                  constants.socialLinks && Object.entries(constants.socialLinks).map(([name, item], index) => {
                    if (item.disable) return null;
                    return (
                      <li key={index}>
                        <a href={item.url} target="_blank" rel='noopener noreferrer' ><i className={`fa ${item.icon}`}></i> {name}</a>
                      </li>
                    )
                  }
                  )
                }
              </ul>
            </div>
          </div>

          <p className="scrolldown">
            <a className="smoothscroll" href="#about"><i className="icon-down-circle"></i></a>
          </p>

        </header>
      </React.Fragment >
    );
  }
}
