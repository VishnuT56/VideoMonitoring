from datetime import datetime

import cv2
import numpy as np
from dat import Database

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image = request.POST['image']















# import the opencv library
import cv2

# define a video capture object
from my_app.models import Emotion

vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    date = datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"

    # Display the resulting frame
    # cv2.imshow('frame', frame)

    cv2.imwrite("C:\\Users\\MY DELL\\PycharmProjects\\video_monitoring\\media\\emotion\\"+date, frame)
    cv2.imshow('frame', frame)

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.load_weights("C:\\Users\\MY DELL\\PycharmProjects\\video_monitoring\\my_app\\model.h5")

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    frame = cv2.imread("C:\\Users\\MY DELL\\PycharmProjects\\video_monitoring\\media\\emotion\\" + date)

    facecasc = cv2.CascadeClassifier(
        "C:\\Users\\MY DELL\\PycharmProjects\\video_monitoring\\my_app\\haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        # print(prediction)
        maxindex = int(np.argmax(prediction))
        print(emotion_dict[maxindex])

        mm = Emotion()
        # lid = request.POST["lid"]
        mm.photo = frame
        mm.emotion = emotion_dict[maxindex]
        mm.date = datetime.now()
        mm.time = datetime.now().time()
        # mm.USER = user.objects.get(LOGIN_id=lid)
        mm.save()

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

# vid = cv2.VideoCapture(0)
# ret, frame = vid.read()
# from datetime import datetime
#
# gg = datetime.now().strftime('%Y%m%d%H%M%S%f') + '.jpg'
# cv2.imwrite("a.jpg", frame)
# cv2.imshow('frame', frame)
#
#
# from datetime import datetime
#
# date = datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"

import base64

# a = base64.b64decode(Image)
# with open("D:\\AIpsychologist\\AIpsychologist\\media\\" + date, "wb") as fh:
#     path = "/media/user/" + date
#     fh.write(a)
#     fh.close()


    # mm = Emotion()
    # lid = request.POST["lid"]
    # mm.Emotion = emotion_dict[maxindex]
    # mm.Date = datetime.now()
    # mm.Time = datetime.now().time()
    # mm.USER = user.objects.get(LOGIN_id=lid)
    # mm.save()
