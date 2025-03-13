import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('image_classification_model11.h5')

# Define the class labels
class_labels = ['cardboard', 'paper', 'metal', 'trash', 'plastic']

# Parameters for image preprocessing
img_width, img_height = 128, 128  # Change to 128x128 to match the trained model

# Function to preprocess the frame
def preprocess_frame(frame):
    # Resize the frame to the required input size of the model
    frame_resized = cv2.resize(frame, (img_width, img_height))
    # Normalize the pixel values
    frame_normalized = frame_resized / 255.0
    # Expand dimensions to match the model's input shape
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    return frame_expanded

# Initialize webcam
cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop for live detection
try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Preprocess the captured frame
        processed_frame = preprocess_frame(frame)

        # Perform prediction
        prediction = model.predict(processed_frame)
        class_index = np.argmax(prediction[0])  # Get the index of the highest probability
        confidence = prediction[0][class_index]  # Get the confidence level

        # Get the label of the predicted class
        predicted_class = class_labels[class_index]
        material_type = "Metal" if predicted_class == "metal" else "Non-Metal"
        label = f"{predicted_class} ({material_type}): {confidence*100:.2f}%"

        # Display the label on the frame
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame with the prediction
        cv2.imshow('Waste Classification', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed.")
