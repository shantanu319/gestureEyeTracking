# gestureEyeTracking
This program combines both and allows for a Window's user to operate their computer--with the exception of typing--hands free. Scrolling, clicking, zooming in and out, as well as navigating to and from different windows on the desktop are all handled by the program. Eye-tracking works the way you would expect it to, and the cursor on the screen follows the direction of the eyes (using the angle of the whites and the pupils). Gesture control is handled with these actions: a pinch to operate sliders horizontally or vertically, a peace sign to move the cursor, pulling one or both index/middle fingers to left and right click, and a flat palm is used as a neutral action to halt all actions. 

Packages used: Numpy, Pyautogui, OpenCV, Mediapipe, Scikit

Currently work is being done to shift primary operation to the eyes (for switching windows, and clicking on objects). The program does not require an extremely high quality camera but it can help, the user should also make sure the background is well lit. 

