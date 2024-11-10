import cv2
import mediapipe as mp

def detect_hand_gesture(mode = 0): 
    # Initialize MediaPipe Hand detector
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Start capturing from the webcam
    cap = cv2.VideoCapture(1)

    # Initialize the hand tracking model
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while True:  # Keep the loop running until the user decides to exit
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video. Exiting...")
                break
        
            # Flip the frame for a mirror view
            frame = cv2.flip(frame, 1)
            # Convert the frame color to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
        
            # Check if hand landmarks are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on the frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                    # Extract landmark positions for gesture recognition
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                
                    # Example of simple gesture detection logic
                    # Gesture 1: Open Hand - Distance between thumb and index finger
                    # ***The coordinates are reverse along x-axis(I don't know why...)
                    if thumb_tip.y > index_tip.y and index_tip.y > middle_tip.y:
                        gesture_text = "Open Hand"
                    else:
                        gesture_text = "Unknown Gesture"

                    coordinates_text = f"Thumb Tip Y: {thumb_tip.y:.4f} | Index Tip Y: {index_tip.y:.4f} | Middle Tip Y: {middle_tip.y:.4f}"
                    if mode == 1:
                        # Display the gesture name on the frame
                        cv2.putText(frame, gesture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        # Display the y-coordinates below the gesture text
                        cv2.putText(frame, coordinates_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


            # Show the frame
            cv2.imshow('Hand Gesture Recognition', frame)
        
            # Wait for the 'q' key to be pressed to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting the program...")
                break

    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()

detect_hand_gesture(1)
