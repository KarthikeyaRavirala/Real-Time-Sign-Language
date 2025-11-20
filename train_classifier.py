import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("Loading data from pickle file...")
try:
    # Load data from the pickle file
    data_dict = pickle.load(open('./data.pickle', 'rb'))
    print("Data loaded successfully!")
except Exception as e:
    print("Error loading data:", e)
    exit(1)

try:
    # Extract data and labels
    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
except Exception as e:
    print("Error processing data:", e)
    exit(1)

try:
    # Flatten the data and ensure landmarks are structured as arrays
    data_flattened = []
    for d in data:
        flattened_landmarks = np.concatenate([landmark.reshape(-1) for landmark in d])
        data_flattened.append(flattened_landmarks)

    # Convert the flattened data to a numpy array
    data_flattened = np.array(data_flattened)
    print(f"Flattened data shape: {data_flattened.shape}")
except Exception as e:
    print("Error flattening data:", e)
    exit(1)

try:
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data_flattened, labels, test_size=0.2, shuffle=True, stratify=labels)
    print("Data split successfully!")
except Exception as e:
    print("Error splitting data:", e)
    exit(1)

try:
    # Initialize the RandomForestClassifier
    model = RandomForestClassifier()
    print("Model initialized!")
except Exception as e:
    print("Error initializing model:", e)
    exit(1)

try:
    # Train the model
    print("Training the model...")
    model.fit(x_train, y_train)
    print("Model trained successfully!")
except Exception as e:
    print("Error training model:", e)
    exit(1)

try:
    # Make predictions
    y_predict = model.predict(x_test)
    print("Predictions made!")
except Exception as e:
    print("Error making predictions:", e)
    exit(1)

try:
    # Calculate accuracy
    score = accuracy_score(y_predict, y_test)
    print('{}% of samples were classified correctly!'.format(score * 100))
except Exception as e:
    print("Error calculating accuracy:", e)
    exit(1)

try:
    # Save the trained model
    with open('model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
    print("Model saved successfully!")
except Exception as e:
    print("Error saving model:", e)
    exit(1)