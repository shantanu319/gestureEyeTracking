# Imports

import cv2
import mediapipe as mp
import pyautogui
import math
import platform
import sys
import time
from enum import IntEnum
# Add platform check for Windows-specific imports
if platform.system() == 'Windows':
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# For macOS, use a different approach for volume control
elif platform.system() == 'Darwin':  # Darwin is the name of the macOS core
    import subprocess
    import re
from google.protobuf.json_format import MessageToDict
import screen_brightness_control as sbcontrol

pyautogui.FAILSAFE = False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# Gesture Encodings
class Gest(IntEnum):
    # Binary Encoded
    """
    Enum for mapping all hand gesture to binary number.
    """

    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    LAST4 = 15
    THUMB = 16
    PALM = 31

    # Extra Mappings
    V_GEST = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36


# Multi-handedness Labels
class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1


# Convert Mediapipe Landmarks to recognizable Gestures
class HandRecog:
    """
    Convert Mediapipe Landmarks to recognizable Gestures.
    """

    def __init__(self, hand_label):
        self.finger = 0
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.hand_result = None
        self.hand_label = hand_label

    def update_hand_result(self, hand_result):
        self.hand_result = hand_result

    def get_signed_dist(self, point):
        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x) ** 2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y) ** 2
        dist = math.sqrt(dist)
        return dist * sign

    def get_dist(self, point):
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x) ** 2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y) ** 2
        dist = math.sqrt(dist)
        return dist

    def get_dz(self, point):
        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)

    # Function to find Gesture Encoding using current finger_state.
    # Finger_state: 1 if finger is open, else 0
    def set_finger_state(self):
        if self.hand_result == None:
            return

        points = [[8, 5, 0], [12, 9, 0], [16, 13, 0], [20, 17, 0]]
        self.finger = 0
        self.finger = self.finger | 0  # thumb
        for idx, point in enumerate(points):

            dist = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])

            try:
                ratio = round(dist / dist2, 1)
            except:
                ratio = round(dist1 / 0.01, 1)

            self.finger = self.finger << 1
            if ratio > 0.5:
                self.finger = self.finger | 1

    # Handling Fluctations due to noise
    def get_gesture(self):
        if self.hand_result == None:
            return Gest.PALM

        current_gesture = Gest.PALM
        if self.finger in [Gest.LAST3, Gest.LAST4] and self.get_dist([8, 4]) < 0.05:
            if self.hand_label == HLabel.MINOR:
                current_gesture = Gest.PINCH_MINOR
            else:
                current_gesture = Gest.PINCH_MAJOR

        elif Gest.FIRST2 == self.finger:
            point = [[8, 12], [5, 9]]
            dist1 = self.get_dist(point[0])
            dist2 = self.get_dist(point[1])
            ratio = dist1 / dist2
            if ratio > 1.7:
                current_gesture = Gest.V_GEST
            else:
                if self.get_dz([8, 12]) < 0.1:
                    current_gesture = Gest.TWO_FINGER_CLOSED
                else:
                    current_gesture = Gest.MID

        else:
            current_gesture = self.finger

        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.prev_gesture = current_gesture

        if self.frame_count > 4:
            self.ori_gesture = current_gesture
        return self.ori_gesture


# Executes commands according to detected gestures
class Controller:
    """
    Commands:
    tx_old : int
        previous mouse location x coordinate
    ty_old : int
        previous mouse location y coordinate
    flag : bool
        true if V gesture is detected
    grabflag : bool
        true if FIST gesture is detected
    pinchmajorflag : bool
        true if PINCH gesture is detected through MAJOR hand,
        on x-axis 'Controller.changesystembrightness',
        on y-axis 'Controller.changesystemvolume'.
    pinchminorflag : bool
        true if PINCH gesture is detected through MINOR hand,
        on x-axis 'Controller.scrollHorizontal',
        on y-axis 'Controller.scrollVertical'.
    pinchstartxcoord : int
        x coordinate of hand landmark when pinch gesture is started.
    pinchstartycoord : int
        y coordinate of hand landmark when pinch gesture is started.
    pinchdirectionflag : bool
        true if pinch gesture movment is along x-axis,
        otherwise false
    prevpinchlv : int
        stores quantized magnitued of prev pinch gesture displacment, from
        starting position
    pinchlv : int
        stores quantized magnitued of pinch gesture displacment, from
        starting position
    framecount : int
        stores no. of frames since 'pinchlv' is updated.
    prev_hand : tuple
        stores (x, y) coordinates of hand in previous frame.
    pinch_threshold : float
        step size for quantization of 'pinchlv'.
    """

    tx_old = 0
    ty_old = 0
    trial = True
    flag = False
    grabflag = False
    pinchmajorflag = False
    pinchminorflag = False
    pinchstartxcoord = None
    pinchstartycoord = None
    pinchdirectionflag = None
    prevpinchlv = 0
    pinchlv = 0
    framecount = 0
    prev_hand = None
    pinch_threshold = 0.3
    
    if platform.system() == 'Windows':
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
        except Exception as e:
            print(f"Error initializing Windows audio control: {e}")
            volume = None
    elif platform.system() == 'Darwin':
        volume = None
    
    @classmethod
    def getpinchylv(cls, hand_result):
        dist = round((cls.pinchstartycoord - hand_result.landmark[8].y) * 10, 1)
        return dist

    @classmethod
    def getpinchxlv(cls, hand_result):
        dist = round((hand_result.landmark[8].x - cls.pinchstartxcoord) * 10, 1)
        return dist

    @classmethod
    def changesystembrightness(cls):
        currentBrightnessLv = sbcontrol.get_brightness(display=0) / 100.0
        currentBrightnessLv += cls.pinchlv / 50.0
        if currentBrightnessLv > 1.0:
            currentBrightnessLv = 1.0
        elif currentBrightnessLv < 0.0:
            currentBrightnessLv = 0.0
        sbcontrol.fade_brightness(int(100 * currentBrightnessLv), start=sbcontrol.get_brightness(display=0))

    @classmethod
    def changesystemvolume(cls):
        if platform.system() == 'Windows':
            if cls.volume:
                try:
                    volpercent = 1 - cls.pinchlv/10
                    volpercent = max(0.0, min(1.0, volpercent))  # Ensure it's between 0 and 1
                    cls.volume.SetMasterVolumeLevelScalar(volpercent, None)
                except Exception as e:
                    print(f"Error changing Windows volume: {e}")
        elif platform.system() == 'Darwin':
            try:
                volpercent = int(100 - (cls.pinchlv * 10))
                volpercent = max(0, min(100, volpercent))  # Ensure it's between 0 and 100
                subprocess.call(["osascript", "-e", f"set volume output volume {volpercent}"])
            except Exception as e:
                print(f"Error changing macOS volume: {e}")

    @classmethod
    def scrollVertical(cls):
        pyautogui.scroll(120 if cls.pinchlv > 0.0 else -120)

    @classmethod
    def scrollHorizontal(cls):
        pyautogui.keyDown('shift')
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(-120 if cls.pinchlv > 0.0 else 120)
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('shift')

    @classmethod
    def get_position(cls, hand_result):
        point = 9
        position = [hand_result.landmark[point].x, hand_result.landmark[point].y]
        sx, sy = pyautogui.size()
        x_old, y_old = pyautogui.position()
        x = int(position[0] * sx)
        y = int(position[1] * sy)
        if cls.prev_hand is None:
            cls.prev_hand = x, y
        delta_x = x - cls.prev_hand[0]
        delta_y = y - cls.prev_hand[1]

        distsq = delta_x ** 2 + delta_y ** 2
        
        # Reduce damping/smoothing for more responsive movement
        ratio = 1
        cls.prev_hand = [x, y]

        # Adjust sensitivity threshold for smoother but more responsive movement
        if distsq <= 25:
            ratio = 0
        elif distsq <= 900:  # Increased from a potential lower value
            ratio = 0.07 * (distsq ** (1/2))
        else:
            ratio = 0.7  # Increased from 0.5 for more responsive movement
        
        x, y = x_old + delta_x * ratio, y_old + delta_y * ratio
        return (x, y)

    @classmethod
    def pinch_control_init(cls, hand_result):
        cls.pinchstartxcoord = hand_result.landmark[8].x
        cls.pinchstartycoord = hand_result.landmark[8].y
        cls.pinchlv = 0
        cls.prevpinchlv = 0
        cls.framecount = 0
        
    @classmethod
    def pinch_control(cls, hand_result, controlHorizontal, controlVertical):
        if cls.framecount == 5:
            cls.framecount = 0
            cls.pinchlv = cls.prevpinchlv

            if cls.pinchdirectionflag == True:
                controlHorizontal()  # x

            elif cls.pinchdirectionflag == False:
                controlVertical()  # y

        lvx = cls.getpinchxlv(hand_result)
        lvy = cls.getpinchylv(hand_result)

        if abs(lvy) > abs(lvx) and abs(lvy) > cls.pinch_threshold:
            cls.pinchdirectionflag = False
            if abs(cls.prevpinchlv - lvy) < cls.pinch_threshold:
                cls.framecount += 1
            else:
                cls.prevpinchlv = lvy
                cls.framecount = 0

        elif abs(lvx) > cls.pinch_threshold:
            cls.pinchdirectionflag = True
            if abs(cls.prevpinchlv - lvx) < cls.pinch_threshold:
                cls.framecount += 1
            else:
                cls.prevpinchlv = lvx
                cls.framecount = 0

    @classmethod
    def handle_controls(cls, gesture, hand_result):
        x, y = None, None
        if gesture != Gest.PALM:
            x, y = cls.get_position(hand_result)

        # flag reset
        if gesture != Gest.FIST and cls.grabflag:
            cls.grabflag = False
            pyautogui.mouseUp(button="left")

        if gesture != Gest.PINCH_MAJOR and cls.pinchmajorflag:
            cls.pinchmajorflag = False

        if gesture != Gest.PINCH_MINOR and cls.pinchminorflag:
            cls.pinchminorflag = False

        # implementation
        if gesture == Gest.V_GEST:
            cls.flag = True
            # Reduce the duration to make movement more immediate (0.1 -> 0.05 or remove entirely)
            pyautogui.moveTo(x, y)

        elif gesture == Gest.FIST:
            if not cls.grabflag:
                cls.grabflag = True
                pyautogui.mouseDown(button="left")
            pyautogui.moveTo(x, y, duration=0.1)

        elif gesture == Gest.MID and cls.flag:
            pyautogui.click()
            cls.flag = False

        elif gesture == Gest.INDEX and cls.flag:
            pyautogui.click(button='right')
            cls.flag = False

        elif gesture == Gest.TWO_FINGER_CLOSED and cls.flag:
            pyautogui.doubleClick()
            cls.flag = False

        elif gesture == Gest.PINCH_MINOR:
            if cls.pinchminorflag == False:
                cls.pinch_control_init(hand_result)
                cls.pinchminorflag = True
            cls.pinch_control(hand_result, cls.scrollHorizontal, cls.scrollVertical)

        elif gesture == Gest.PINCH_MAJOR:
            if cls.pinchmajorflag == False:
                cls.pinch_control_init(hand_result)
                cls.pinchmajorflag = True
            cls.pinch_control(hand_result, cls.changesystembrightness, cls.changesystemvolume)

class GestureController:
    gc_mode = 0
    cap = None
    CAM_HEIGHT = None
    CAM_WIDTH = None
    hr_major = None  # Right Hand by default
    hr_minor = None  # Left hand by default
    dom_hand = True

    def __init__(self):
        """Initilaizes attributes."""
        GestureController.gc_mode = 1
        GestureController.cap = cv2.VideoCapture(0)
        
        # Increase camera buffer size to reduce lag
        GestureController.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set camera resolution to a lower value for better performance
        GestureController.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        GestureController.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Set camera FPS to higher value if supported
        GestureController.cap.set(cv2.CAP_PROP_FPS, 30)
        
        GestureController.CAM_HEIGHT = GestureController.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        GestureController.CAM_WIDTH = GestureController.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    @classmethod
    def classify_hands(cls, results):
        if GestureController.hr_major == None:
            GestureController.hr_major = HandRecog(HLabel.MAJOR)

        if GestureController.hr_minor == None:
            GestureController.hr_minor = HandRecog(HLabel.MINOR)

        handedness_dict = []
        if results.multi_handedness:
            for idx, hand_handedness in enumerate(results.multi_handedness):
                handedness_dict.append({
                    'index': idx,
                    'score': hand_handedness.classification[0].score,
                    'label': hand_handedness.classification[0].label
                })

        if len(handedness_dict) == 2:
            # Any hand can be dominant in this case
            if handedness_dict[0]['label'] == 'Left':
                if GestureController.dom_hand:
                    GestureController.hr_major.update_hand_result(results.multi_hand_landmarks[0])
                    GestureController.hr_minor.update_hand_result(results.multi_hand_landmarks[1])
                else:
                    GestureController.hr_major.update_hand_result(results.multi_hand_landmarks[1])
                    GestureController.hr_minor.update_hand_result(results.multi_hand_landmarks[0])
            else:
                if GestureController.dom_hand:
                    GestureController.hr_major.update_hand_result(results.multi_hand_landmarks[1])
                    GestureController.hr_minor.update_hand_result(results.multi_hand_landmarks[0])
                else:
                    GestureController.hr_major.update_hand_result(results.multi_hand_landmarks[0])
                    GestureController.hr_minor.update_hand_result(results.multi_hand_landmarks[1])

        else:
            # Checking if we have at least one hand
            if len(handedness_dict) == 1:
                if handedness_dict[0]['label'] == 'Right':
                    if GestureController.dom_hand:
                        GestureController.hr_major.update_hand_result(results.multi_hand_landmarks[0])
                    else:
                        GestureController.hr_minor.update_hand_result(results.multi_hand_landmarks[0])
                else:
                    if GestureController.dom_hand:
                        GestureController.hr_minor.update_hand_result(results.multi_hand_landmarks[0])
                    else:
                        GestureController.hr_major.update_hand_result(results.multi_hand_landmarks[0])

    def start(self):
        handmajor = HandRecog(HLabel.MAJOR)
        handminor = HandRecog(HLabel.MINOR)

        cv2.namedWindow("Gesture Controller", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gesture Controller", 640, 480)
        
        cv2.createTrackbar("Min Detection Confidence", "Gesture Controller", 50, 100, lambda x: None)
        cv2.createTrackbar("Min Tracking Confidence", "Gesture Controller", 50, 100, lambda x: None)

        prev_frame_time = 0
        new_frame_time = 0
        
        min_detection_conf = 0.5
        min_tracking_conf = 0.5
        
        # Initialize with static_image_mode=False for better tracking performance
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf)

        while GestureController.gc_mode:
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time) if (new_frame_time-prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time
            
            # Only update detection parameters if they've changed significantly to avoid recreating the model
            new_min_detection_conf = cv2.getTrackbarPos("Min Detection Confidence", "Gesture Controller") / 100.0
            new_min_tracking_conf = cv2.getTrackbarPos("Min Tracking Confidence", "Gesture Controller") / 100.0
            
            significant_change = abs(new_min_detection_conf - min_detection_conf) > 0.05 or abs(new_min_tracking_conf - min_tracking_conf) > 0.05
            
            if significant_change:
                min_detection_conf = new_min_detection_conf
                min_tracking_conf = new_min_tracking_conf
                hands.close()
                hands = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=min_detection_conf,
                    min_tracking_confidence=min_tracking_conf)
            
            success, image = GestureController.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            image = cv2.flip(image, 1)
            
            # Process a smaller image for faster detection
            small_image = cv2.resize(image, (320, 240))
            small_image.flags.writeable = False
            small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
            results = hands.process(small_image)
            
            # Using the original image only for drawing
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Skip drawing in every frame for higher performance
            should_draw = int(fps) % 2 == 0  # Only draw every other frame
            
            if should_draw:
                cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"Detection conf: {min_detection_conf:.2f}", (10, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(image, f"Tracking conf: {min_tracking_conf:.2f}", (10, 230), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if not results.multi_hand_landmarks:
                if should_draw:
                    cv2.putText(image, "Tips for better detection:", (10, 280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.putText(image, "- Ensure good lighting", (30, 310), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
                    cv2.putText(image, "- Keep hand in frame", (30, 340), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
                    cv2.putText(image, "- Lower detection confidence", (30, 370), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
                    cv2.putText(image, "- Try different hand positions", (30, 400), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
        
            if results.multi_hand_landmarks:
                if should_draw:
                    cv2.putText(image, f"Hands detected: {len(results.multi_hand_landmarks)}", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Simplified landmark drawing for better performance
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
            else:
                if should_draw:
                    cv2.putText(image, "No hands detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(image, "Try adjusting lighting or camera position", (10, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Process gesture recognition even if we're not drawing UI
            if results.multi_hand_landmarks:
                GestureController.classify_hands(results)
                handmajor.update_hand_result(GestureController.hr_major.hand_result)
                handminor.update_hand_result(GestureController.hr_minor.hand_result)

                handmajor.set_finger_state()
                handminor.set_finger_state()

                # Process dominant hand first
                gest_name = handmajor.get_gesture()
                if gest_name == Gest.PINCH_MAJOR:
                    Controller.handle_controls(gest_name, handmajor.hand_result)
                else:
                    gest_name = handminor.get_gesture()
                    if gest_name == Gest.PINCH_MINOR:
                        Controller.handle_controls(gest_name, handminor.hand_result)
                    else:
                        gest_name = handmajor.get_gesture()
                        Controller.handle_controls(gest_name, handmajor.hand_result)
                        
                # Display current gesture if drawing the frame
                if should_draw:
                    if gest_name is not None:
                        try:
                            gest_str = str(gest_name).split('.')[-1]
                        except:
                            gest_str = str(gest_name)
                    else:
                        gest_str = "NONE"
                    cv2.putText(image, f"Gesture: {gest_str}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                Controller.prev_hand = None
        
            # Only show the frame at a reasonable frame rate to avoid blocking CPU
            cv2.imshow('Gesture Controller', image)
            
            # Use a shorter wait time for key checking
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
                break

        hands.close()
        GestureController.cap.release()
        cv2.destroyAllWindows()

gc1 = GestureController()
gc1.start()
