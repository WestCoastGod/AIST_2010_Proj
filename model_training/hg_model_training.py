import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def model_training():
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'gesture_data.csv')
    # Load dataset
    data = pd.read_csv(file_path)
    X = data.drop('gesture', axis=1)
    y = data['gesture']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    model_file_path = os.path.join(script_dir, 'gesture_model.pkl')
    # Save the model
    import joblib
    joblib.dump(model, model_file_path)
    print("Place gesture_model.pkl at the same folder of detect_hand_gesture.py")

model_training()
