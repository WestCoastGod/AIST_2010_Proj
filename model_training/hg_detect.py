import cv2
import mediapipe as mp
import joblib
import numpy as np
import os

# Load the trained model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(script_dir, 'gesture_model.pkl')
model = joblib.load(model_file_path)
gestures = ["open_hand", "fist", "pointing"]

# Initialize MediaPipe Hand detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)

                landmarks = np.array(landmarks).reshape(1, -1)

                # Predict the gesture
                gesture_id = model.predict(landmarks)[0]
                gesture_text = gestures[int(gesture_id)]

                cv2.putText(frame, gesture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()