import React, { useEffect, useRef, useState } from 'react';
import './index.css';
import { NavLink } from 'react-router-dom';
import SearchComponent from '../../Components/Search/Search.component';
import '../../assets/bootstrap/bootstrap.min.css';
import CanvasDraw from 'react-canvas-draw';
import logo2 from '../../assets/logo/logo2.png';
import pencil from '../../assets/logo/pencil.png';
import searchI from '../../assets/logo/search.png';
import triangle from '../../assets/logo/obj_triangle.png';
import axios from 'axios';

function HomeScreen() {
  const canvasRef = useRef(null);
  const canvas = '#ffff66';
  const brush = 4;
  let dataFinal;
  const [isShown, setIsShown] = useState(false);
  const [draw, setDraw] = useState();
  const [result, setResult] = useState([]);
  const handleClick = (event) => {
    setIsShown((current) => !current);
  };
  const [search, setSearch] = useState('');
  const onChangeDraw = async () => {
    if (canvasRef.current) {
      let subData = canvasRef.current.canvasContainer.children[1].toDataURL();
      dataFinal = subData.substring(22);
      console.log(dataFinal);

      const form = new FormData();
      form.append('data',dataFinal)
      const res = await axios.post('//localhost:5000/model', form, {
       headers: {
      'Content-Type': 'multipart/form-data'
       }
      }).then(function (response) {
        //handle success
        console.log(response);
      })
      .catch(function (response) {
        //handle error
        console.log(response);
      });
      //const res = await axios.post(`http://localhost:5000/model`, { "data": dataFinal });
      setResult(res);
      console.log(res);
    }
  };

  return (
    <div>
      <div className='jumbotron d-flex align-items-center'>
        <img src={draw} alt='' />
        <div className='container text-center'>
          <img src={logo2} alt='logo' className='logo2' />
          <div className='input-method'>
            <div className='input-method'>
              <div className='input-method-button' onClick={handleClick}>
                <img src={pencil} alt='' className='input-icon' />
              </div>
              <div className='search-wrapper'>
                <input
                  type='text'
                  placeholder='Tra hán tự: hán, 漢, かん'
                  id='inputWord'
                  value={search}
                  onChange={(event) => {
                    setSearch(event.target.value);
                  }}
                />
                <NavLink to={`/detail/${search}`}>
                  <img src={searchI} alt='' className='searchI' />
                </NavLink>
              </div>
              <div className='draw-writing'>
                {isShown && (
                  <div className='hand-writing-area-wrapper'>
                    <div className='draw-kanji-top-bar'>
                      <div className='draw-kanji-result'>result</div>
                      <div className='writing-button-area'>
                        <div>
                          <button
                            onClick={() => {
                              canvasRef.current.undo();
                            }}
                          >
                            <i class='fa fa-repeat icon'></i>
                          </button>
                          <button
                            onClick={() => {
                              canvasRef.current.clear();
                            }}
                          >
                            <i class='fa fa-eraser icon'></i>
                          </button>
                          <button
                            onClick={() => {
                              handleClick(false);
                            }}
                          >
                            <i class='fa fa-window-close icon'></i>
                          </button>
                        </div>
                      </div>
                      <div className='hand-writing-area margin-btm'>
                        <div className='canvas-draw-kanji'>
                          <CanvasDraw
                            ref={canvasRef}
                            onChange={onChangeDraw}
                            brushColor={canvas}
                            brushRadius={brush}
                            hideGrid={true}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
        <div className='rectangle-1'></div>
        <div className='rectangle-2'></div>
        <div className='rectangle-transparent-1'></div>
        <div className='rectangle-transparent-2'></div>
        <div className='circle-1'></div>
        <div className='circle-2'></div>
        <div className='circle-3'></div>
        <div className='triangle triangle-1'>
          <img src={triangle} alt='' />
        </div>
        <div className='triangle triangle-2'>
          <img src={triangle} alt='' />
        </div>
        <div className='triangle triangle-3'>
          <img src={triangle} alt='' />
        </div>
        <div className='triangle triangle-4'>
          <img src={triangle} alt='' />
        </div>
      </div>
    </div>
  );
}

export default HomeScreen;
