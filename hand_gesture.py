import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier   # Faster than SVM
from sklearn.metrics import accuracy_score
import pickle

# Dataset path
dataset_path = "leapGestRecog"

images = []
labels = []

print("Loading dataset...")

max_images_per_class = 100   # LIMIT images (very important)

# Loop through person folders (00–09)
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if os.path.isdir(person_path):

        # Loop through gesture folders
        for gesture in os.listdir(person_path):
            gesture_path = os.path.join(person_path, gesture)

            if os.path.isdir(gesture_path):

                count = 0  # counter for limiting images

                # Loop through images
                for image_name in os.listdir(gesture_path):

                    if count >= max_images_per_class:
                        break

                    image_path = os.path.join(gesture_path, image_name)
                    img = cv2.imread(image_path)

                    if img is not None:
                        img = cv2.resize(img, (32, 32))  # smaller = faster
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = img.flatten()

                        images.append(img)
                        labels.append(gesture)

                        count += 1

print("Total images loaded:", len(images))

# Convert to numpy arrays
X = np.array(images)
y = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Training model...")

# Faster classifier
model = SGDClassifier(max_iter=1000)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy * 100, "%")

# Save model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as gesture_model.pkl")