# Import necessary modules
import pickle  # Module for serializing and deserializing Python objects
import cv2  # OpenCV for video capture and image processing
import mediapipe as mp  # MediaPipe for hand detection and landmark processing
import numpy as np  # NumPy for array and numerical operations
import warnings
warnings.filterwarnings("ignore")

# Global variables for models
model = None
int_to_label = None
TF_AVAILABLE = False

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available, using fallback method")

# Load the pre-trained model
if TF_AVAILABLE:
    try:
        model = load_model('lstm_model.h5')
        print("LSTM model loaded successfully!")
    except Exception as e:
        print("Error loading LSTM model:", e)
        model = None

    # Load label mapping
    try:
        label_dict = pickle.load(open('./label_mapping.p', 'rb'))
        int_to_label = label_dict['int_to_label']
        print("Label mapping loaded successfully!")
    except Exception as e:
        print("Error loading label mapping:", e)
        int_to_label = None
else:
    # Fallback to original model
    try:
        model_dict = pickle.load(open('./model.p', 'rb'))
        model = model_dict['model']
        print("Original model loaded successfully!")
    except Exception as e:
        print("Error loading original model:", e)
        model = None

# Initialize video capture from the default camera (index 0)
cap = cv2.VideoCapture(0)

# Initialize MediaPipe's hand detection and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure MediaPipe Hands for static image processing
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define a dictionary for mapping model output to sign labels (for fallback)
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello', 
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please', 32: 'You are welcome.'
}

while True:
    data_aux = []  # Auxiliary list to store normalized landmark coordinates
    x_ = []  # List to store x-coordinates of landmarks
    y_ = []  # List to store y-coordinates of landmarks

    ret, frame = cap.read()  # Capture a frame from the video feed
    if not ret:
        break
        
    H, W, _ = frame.shape  # Get the dimensions of the frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB format

    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)
    
    # Initialize bounding box coordinates
    x1, y1, x2, y2 = 0, 0, 0, 0
    
    if results.multi_hand_landmarks:
        # Draw hand landmarks and connections on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            # Extract x and y coordinates of each landmark
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Normalize landmark coordinates and add to data_aux
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Calculate bounding box coordinates for the hand landmarks
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

    try:
        if len(data_aux) > 0:
            if TF_AVAILABLE and model is not None and int_to_label is not None:
                # Reshape data for LSTM model
                # Convert to numpy array and reshape to (1, timesteps, features)
                n_features = 2  # x and y coordinates
                n_timesteps = len(data_aux) // n_features
                if n_timesteps > 0:
                    data_reshaped = np.array(data_aux).reshape(1, n_timesteps, n_features)
                    
                    # Predict using the LSTM model
                    prediction = model.predict(data_reshaped, verbose=0)
                    predicted_class = np.argmax(prediction)
                    confidence = np.max(prediction)
                    predicted_character = int_to_label.get(predicted_class, "Unknown")
                    
                    print(f"Predicted character: {predicted_character} (Confidence: {confidence:.2f})")

                    # Draw a bounding box and label around the detected hand
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, f"{predicted_character} ({confidence:.2f})", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            elif not TF_AVAILABLE and model is not None:
                # Fallback to original model
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                print("Predicted character : ", predicted_character)

                # Draw a bounding box and label around the detected hand
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    except Exception as e:
        # Handle exceptions that might occur during prediction
        print("Error during prediction:", e)
        pass

    # Display the frame in a window
    cv2.imshow('frame', frame)

    # Check if the window was closed by the user (pressing 'q' key)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()