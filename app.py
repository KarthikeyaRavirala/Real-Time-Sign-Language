# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt

from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Global variables for sentence building
sentence_building = False
word_building = False
current_word = ""
current_sentence = ""
last_predicted_character = ""
mode = "continuous"  # "continuous" or "step"

# Try to load the model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading the model:", e)
    model = None

# Define labels dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello',
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please',
    32: 'You are welcome.'
}

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('start_sentence')
def handle_start_sentence():
    global sentence_building, current_sentence
    sentence_building = True
    current_sentence = ""
    emit('sentence_started')
    emit('prediction', {'sentence': current_sentence, 'current_word': current_word})

@socketio.on('end_sentence')
def handle_end_sentence():
    global sentence_building, current_sentence
    sentence_building = False
    emit('sentence_ended')
    emit('prediction', {'sentence': current_sentence, 'current_word': current_word})

@socketio.on('start_word')
def handle_start_word():
    global word_building, current_word
    word_building = True
    current_word = ""
    emit('word_started')
    emit('prediction', {'sentence': current_sentence, 'current_word': current_word})

@socketio.on('end_word')
def handle_end_word():
    global word_building, current_word, current_sentence
    word_building = False
    if current_word:
        if current_sentence:
            current_sentence += " " + current_word
        else:
            current_sentence = current_word
        current_word = ""
    emit('word_ended')
    emit('prediction', {'sentence': current_sentence, 'current_word': current_word})

@socketio.on('reset_sentence')
def handle_reset_sentence():
    global sentence_building, word_building, current_word, current_sentence
    sentence_building = False
    word_building = False
    current_word = ""
    current_sentence = ""
    emit('sentence_reset')
    emit('prediction', {'sentence': current_sentence, 'current_word': current_word})

@socketio.on('next_gesture')
def handle_next_gesture():
    global last_predicted_character
    last_predicted_character = ""
    emit('ready_for_next_gesture')

@socketio.on('set_mode')
def handle_set_mode(data):
    global mode
    mode = data['mode']
    emit('mode_changed', {'mode': mode})

def generate_frames():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    global sentence_building, word_building, current_word, current_sentence, last_predicted_character, mode

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                try:
                    if model is not None:
                        # Predict using the model
                        prediction = model.predict([np.asarray(data_aux)])
                        prediction_proba = model.predict_proba([np.asarray(data_aux)])
                        confidence = max(prediction_proba[0])
                        predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                        
                        # Only update if we have a new prediction and are in the right mode
                        if predicted_character != last_predicted_character or mode == "continuous":
                            last_predicted_character = predicted_character
                            
                            # Add character to current word if word building is active
                            if word_building and predicted_character not in current_word:
                                current_word += predicted_character
                            
                            socketio.emit('prediction', {
                                'text': predicted_character, 
                                'confidence': confidence,
                                'sentence': current_sentence,
                                'current_word': current_word
                            })
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                            cv2.putText(frame, f"{predicted_character} ({confidence*100:.2f}%)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                except Exception as e:
                    print("Prediction error:", e)
                    pass

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001)
