 GPL-3.0 license

# Gaze tracker

This project is an experiment on gaze tracker system, based on the position of glint on the iris, which is the reflection of a source of light located near the camera (See [Purkinje images](https://en.wikipedia.org/wiki/Purkinje_images) and [Eye tracking](https://en.wikipedia.org/wiki/Eye_tracking#Technologies_and_techniques) for details). By tracking the glint it is possible to estimate the gaze position on the screen and use this position as an input system (in this example it is used only to move the mouse arrow). 

## Hardware requirements

<img width=256px align="right" src="https://github.com/luca-ant/gaze_tracker/blob/master/videos/cam.jpg">

For this kind of system, an high resolution image of the eye is required and a source of light near the camera too, so to get that with simple hardware I use a smartphone as a remote camera (achieved by [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) app) fixed on a stable support, turn on the flash LED and put it in front on one eye, close to it. 

<br />

## Demo

### Calibration phase
<p align="center">
  <img width=1024px src="https://github.com/luca-ant/gaze_tracker/blob/master/videos/gaze_tracker_calibration.gif">
</p>

### Mouse control phase
<p align="center">
  <img width=1024px src="https://github.com/luca-ant/gaze_tracker/blob/master/videos/gaze_tracker_mouse.gif">
</p>


## Getting started

* Clone repository
```
git clone https://github.com/luca-ant/gaze_tracker.git
```

* Install dependencies
```
sudo apt install python3-setuptools python3-pip python3-venv
```
or
```
sudo pacman -S python-setuptools python-pip python-virtualenv
```

* Create a virtual environment and install requirement modules
```
cd gaze_tracker
python3 -m venv venv
source venv/bin/activate

python3 -m pip install -r requirements.txt
```

## Running

Simply move to `gt` folder and run `main.py`

```
cd gt
python main.py
```
