import torch
import cv2
import numpy as np
import serial
import time
from collections import Counter
from transformers import AutoModelForImageClassification, AutoProcessor

# Load the model and processor
model_name = "youssefabdelmottaleb/Garbage-Classification-SWIN-Transformer"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# Initialize serial communication with Arduino (Adjust 'COM3' to your Arduino port)
arduino = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)  # Wait for the connection to establish


# Function to preprocess frame
def preprocess_frame(frame):
    """ Preprocess the frame for model inference """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    inputs = processor(images=image, return_tensors="pt")  # Process image
    return inputs


# Open webcam feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize tracking variables
start_time = time.time()
predictions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    inputs = preprocess_frame(frame)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions_tensor = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label_idx = predictions_tensor.argmax().item()
        confidence = predictions_tensor.max().item() * 100  # Convert to percentage

    # Get class names
    labels = model.config.id2label
    predicted_label = labels.get(predicted_label_idx, "Unknown")

    # Store the prediction if confidence is above threshold
    if confidence >= 50:  # Adjust threshold if needed
        predictions.append(predicted_label)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Every 20 seconds, determine the most frequent prediction
    if elapsed_time >= 20:
        if predictions:
            most_common_prediction, count = Counter(predictions).most_common(1)[0]

            # Send the most common prediction to Arduino
            if most_common_prediction in ["plastic", "metal"]:
                arduino.write(b'non_biodegradable\n')
                print("Sent 'non_biodegradable' to Arduino.")
            else:
                arduino.write(b'biodegradable\n')
                print("Sent 'biodegradable' to Arduino.")

            # Reset predictions list
            predictions = []
        start_time = time.time()  # Reset the timer

    # Display results on frame
    label_text = f"Predicted: {predicted_label}, Confidence: {confidence:.2f}%"
    cv2.putText(frame, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Garbage Classification", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()  # Close serial connection to Arduino
print("Webcam released and Arduino connection closed.")