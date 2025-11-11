# ===============================
# 📦 Imports
# ===============================
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ===============================
# 🔹 Load Classifier & Model
# ===============================
face_classifier = cv2.CascadeClassifier(
    r'/Users/pradipkumarde/Desktop/UH Materials/Fall 2025/DS2/facial_emotion_detection/facial_emotion_recognition_ds2/haarcascade_frontalface_default.xml'
)

classifier = load_model(
    r'/Users/pradipkumarde/Desktop/UH Materials/Fall 2025/DS2/facial_emotion_detection/facial_emotion_recognition_ds2/model.h5'
)

# ===============================
# 🔹 Labels
# ===============================
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ===============================
# 🔹 Start Webcam
# ===============================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)  # ← fixed size

        if np.sum(roi_gray) != 0:
            roi = roi_gray.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=-1)  # shape: (48,48,1)
            roi = np.expand_dims(roi, axis=0)   # shape: (1,48,48,1)

            prediction = classifier.predict(roi, verbose=0)[0]
            label = emotion_labels[np.argmax(prediction)]
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
