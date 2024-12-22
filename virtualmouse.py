#Everything Is perfect in this code.
#Left Click,Right Click, Screenshot, Cursor Movement, Volume Up and Down.
#Alignment of Text using CSS is also perfect.
#List is also perfect in html code.

import os   
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pyautogui
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time

app = Flask(__name__)

# Initialize Mediapipe and pyautogui
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

# Scaling factor for smoother cursor movement
smoothing_factor = 3  # Reduced from 5 to increase sensitivity

# Audio volume control initialization using PyCaw
device = AudioUtilities.GetSpeakers()
interface = device.Activate(IAudioEndpointVolume._iid_, 0, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Create the screenshot folder if it doesn't exist
screenshot_folder = "screenshot"
if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)

def change_volume(direction):
    """
    Adjust the system volume with increased sensitivity.
    """
    current_volume = volume.GetMasterVolumeLevelScalar()
    new_volume = current_volume + direction
    new_volume = max(0.0, min(new_volume, 1.0))  # Ensure volume is within [0.0, 1.0]
    volume.SetMasterVolumeLevelScalar(new_volume, None)
    print(f"Volume: {new_volume * 100:.0f}%")

@app.route('/')
def index():
    """Route to serve the HTML frontend."""
    return render_template('index.html')

def fingers_open(landmarks):
    """
    Check if the index finger, middle finger, and thumb are open while others are closed.
    """
    index_dist = landmarks[8].y - landmarks[6].y
    middle_dist = landmarks[12].y - landmarks[10].y
    thumb_dist = landmarks[4].x - landmarks[3].x  # Thumb moves horizontally
    ring_closed = landmarks[16].y > landmarks[14].y
    pinky_closed = landmarks[20].y > landmarks[18].y

    return (
        index_dist < -0.02 and
        middle_dist < -0.02 and
        thumb_dist > 0.02 and
        ring_closed and
        pinky_closed
    )

def is_L_shape(landmarks):
    """
    Detect if the thumb and index finger form an L shape, with the other fingers curled.
    """
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    ring_closed = landmarks[16].y > landmarks[14].y
    pinky_closed = landmarks[20].y > landmarks[18].y

    # Check if thumb is extended horizontally and index is extended upwards (forming L)
    thumb_index_dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    thumb_angle = thumb_tip.x < index_tip.x and thumb_index_dist < 0.1  # Thumb pointing left, close distance

    return thumb_angle and ring_closed and pinky_closed

def is_fist_with_thumb_left(landmarks):
    """
    Detect if the hand is in a fist with the thumb pointing left.
    """
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    thumb_left = thumb_tip.x < thumb_ip.x < thumb_mcp.x
    fingers_closed = all(
        landmarks[point].y > landmarks[point - 2].y
        for point in range(8, 21, 4)  # Check fingertips of index, middle, ring, and pinky
    )

    return thumb_left and fingers_closed

def is_index_and_thumb_touching(landmarks):
    """
    Detect if the index and thumb are touching for a left-click action.
    """
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Check if the thumb and index finger are close enough to touch
    distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    return distance < 0.05  # Adjust the threshold as needed

def is_three_fingers_extended(landmarks):
    """
    Detect if three fingers are extended for a right-click action.
    """
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    # Check if the index, middle, and ring fingers are extended
    return (
        index_tip.y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y and
        middle_tip.y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and
        ring_tip.y < landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y and
        pinky_tip.y > landmarks[mp_hands.HandLandmark.PINKY_MCP].y  # Pinky is closed
    )

def process_frame():
    """Process the webcam feed for hand gestures."""
    cap = cv2.VideoCapture(0)
    prev_cursor_x, prev_cursor_y = pyautogui.position()
    prev_wrist_y = None  # To track wrist movement
    screenshot_taken = False  # Track if a screenshot has been taken

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            gesture_detected = False  # Flag to check if a gesture is detected

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmarks = hand_landmarks.landmark

                    # Check if only index, middle, and thumb are open
                    if fingers_open(landmarks):
                        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        raw_cursor_x = int(index_tip.x * screen_width)
                        raw_cursor_y = int(index_tip.y * screen_height)

                        # Handle extreme edges
                        if abs(raw_cursor_x - prev_cursor_x) > screen_width // 10:
                            cursor_x = raw_cursor_x
                        else:
                            cursor_x = prev_cursor_x + (raw_cursor_x - prev_cursor_x) // smoothing_factor

                        if abs(raw_cursor_y - prev_cursor_y) > screen_height // 10:
                            cursor_y = raw_cursor_y
                        else:
                            cursor_y = prev_cursor_y + (raw_cursor_y - prev_cursor_y) // smoothing_factor

                        # Constrain cursor within screen boundaries
                        cursor_x = max(0, min(screen_width - 1, cursor_x))
                        cursor_y = max(0, min(screen_height - 1, cursor_y))

                        pyautogui.moveTo(cursor_x, cursor_y)
                        prev_cursor_x, prev_cursor_y = cursor_x, cursor_y
                        cv2.putText(frame, "Cursor Moving", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        gesture_detected = True

                    elif is_L_shape(landmarks):
                        # Take screenshot when L shape is detected
                        if not screenshot_taken:
                            screenshot = pyautogui.screenshot()
                            timestamp = int(time.time())  # Use timestamp to make filenames unique
                            screenshot_path = os.path.join(screenshot_folder, f"screenshot_{timestamp}.png")
                            screenshot.save(screenshot_path)
                            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            screenshot_taken = True
                        gesture_detected = True
                    else:
                        screenshot_taken = False  # Reset if the L shape is not detected

                    if is_fist_with_thumb_left(landmarks):
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        wrist_y = wrist.y

                        if prev_wrist_y is None:
                            prev_wrist_y = wrist_y

                        # If wrist moves up (y decreases), increase volume with increased sensitivity
                        if wrist_y < prev_wrist_y:
                            change_volume(0.03)  # Increased volume change
                            cv2.putText(frame, "Volume Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # If wrist moves down (y increases), decrease volume with increased sensitivity
                        elif wrist_y > prev_wrist_y:
                            change_volume(-0.03)  # Increased volume change
                            cv2.putText(frame, "Volume Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        prev_wrist_y = wrist_y
                        gesture_detected = True

                    # Left click detection
                    if is_index_and_thumb_touching(landmarks):
                        pyautogui.click()
                        cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        gesture_detected = True

                    # Right click detection (from code 1)
                    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                    thumb_index_dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
                    thumb_middle_dist = ((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2) ** 0.5

                    if thumb_middle_dist < 0.04 and thumb_index_dist > 0.05:
                        pyautogui.rightClick()
                        cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        gesture_detected = True

            if not gesture_detected:
                cv2.putText(frame, "Gesture Not Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    finally:
        cap.release()

@app.route('/video_feed')
def video_feed():
    """Route to stream video feed."""
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)