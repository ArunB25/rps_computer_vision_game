import imp
from time import time
from tkinter.tix import Tree
import cv2
from keras.models import load_model
import numpy as np
import time
import statistics


if __name__ == '__main__':

    model = load_model('keras_model.h5')
    cap = cv2.VideoCapture(0)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    while True: 
        ret, frame = cap.read()
        frame = cv2.flip(frame,1) # mirror image
        resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
        data[0] = normalized_image
        prediction = model.predict(data)
        
        # Press q to close the window
        print(prediction)
        rps_prediction = np.argmax(prediction)
        print(rps_prediction)


        if rps_prediction == 0:
            guess = "Rock"  
        elif rps_prediction == 1:
            guess = "Paper"
        elif rps_prediction == 2:
            guess = "Scissors"
        elif rps_prediction == 3:
            guess = "Nothing"

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,guess,(55,400), font, 1, (10, 10, 10), 4, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    