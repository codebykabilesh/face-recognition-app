import cv2
cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
    ret,frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Better settings to reduce background detection
        faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
        for i,face in enumerate(faces):
            x,y,w,h = face
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0.0,225),4)
            label = f"Face {i + 1}"
            cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_DUPLEX,0.8,(0,0,225),2)
        cv2.imshow("My window",frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()


