import cv2
from keras.models import load_model
import numpy as np


def get_prediction():

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
        cv2.imshow('frame', frame)
        # Press q to close the window
        print(prediction)
        rps_prediction = np.argmax(prediction)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        if rps_prediction == 0:
            print("user guessed Rock")
            return("Rock")
        elif rps_prediction == 1:
            print("user guessed Paper")
            return("Paper")
        elif rps_prediction == 2:
            print("user guessed Paper")
            return("scissors")
        elif rps_prediction == 3:
            return("nothing guessed, show hand sign for Rock, Paper or Scissors")
        else:
            return("ERROR no highest prediction")


                
    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("start")
    prediction = get_prediction()
    print(prediction)