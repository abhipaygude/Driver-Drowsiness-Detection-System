import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from playsound import playsound

# Eye Aspect Ratio (EAR) calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load face detector & shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 20
counter = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        cv2.polylines(frame, [leftEye], True, (0,255,0), 1)
        cv2.polylines(frame, [rightEye], True, (0,255,0), 1)

        if ear < EAR_THRESHOLD:
            counter += 1
            if counter >= FRAME_THRESHOLD:
                cv2.putText(frame, "DROWSINESS ALERT 🚨", (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                playsound("alert.wav")  # play alarm sound
        else:
            counter = 0

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
