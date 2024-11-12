import cv2
import mediapipe as mp
import csv
import os

def data_collection():
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    # Initialize MediaPipe Hand detector
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Start capturing from the webcam
    cap = cv2.VideoCapture(0)

    # Labels for your gestures
    gestures = ["open_hand", "fist"]
    current_gesture = 0
    data = []

    # Initialize the hand tracking model
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video. Exiting...")
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            readme = 'Press c to capture landmarks, n to switch to the next gesture, q to quit'
            cv2.putText(frame, readme , (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Display the current gesture label
            cv2.putText(frame, f"Gesture: {gestures[current_gesture]}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Extract landmark positions as features
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)
                    
                    # Press 'c' to collect data for the current gesture
                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        data.append([*landmarks, current_gesture])
                        print(f"Collected data for gesture: {gestures[current_gesture]}")
            
            # Change gesture label with keys
            if cv2.waitKey(1) & 0xFF == ord('n'):  # Press 'n' to move to the next gesture
                current_gesture = (current_gesture + 1) % len(gestures)
            
            # Show the frame
            cv2.imshow('Data Collection', frame)

            # Break the loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'gesture_data.csv')
    # Save collected data to CSV
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x' + str(i) for i in range(42)] + ['gesture'])
        writer.writerows(data)

    cap.release()
    cv2.destroyAllWindows()

data_collection()
