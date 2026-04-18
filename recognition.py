import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

names = ['Unknown', 'Naija']  # Add names according to IDs

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if confidence < 100:
            name = names[id]
        else:
            name = "Unknown"

        cv2.putText(img, str(name), (x+5,y-5), font, 1, (255,255,255), 2)

    cv2.imshow('Face Recognition', img)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
