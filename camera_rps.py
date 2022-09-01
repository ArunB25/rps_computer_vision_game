import imp
from time import time
import cv2
from keras.models import load_model
import numpy as np
import time
import statistics
import random

def play():
    def get_prediction():
        '''
        takes images from the camera for 2.5 seconds and returns the mode guess from the machine learning Rock,Paper,Scissors model
        '''

        model = load_model('keras_model.h5')
        cap = cv2.VideoCapture(0)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        ret, frame = cap.read()
        frame = cv2.flip(frame,1) # mirror image
        resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
        data[0] = normalized_image
        prediction = model.predict(data)
        cv2.imshow('frame', frame)
        #print(prediction)
        rps_prediction = np.argmax(prediction)

        cap.release()
        # Destroy all the windows
        cv2.destroyAllWindows()

        print(rps_prediction)
        if rps_prediction == 0:
            return("rock")
        elif rps_prediction == 1:
            return("paper")
        elif rps_prediction == 2:
            return("scissors")
        elif rps_prediction == 3:
            print("nothing guessed, show hand sign for Rock, Paper or Scissors")
            return("nothing")
        else:
            return("ERROR no highest prediction")


    def get_computer_choice():
            '''
            Returns random value of rock, paper or scissors
            '''
            options = ["rock","paper","scissors"]
            return random.choice(options)


    def get_winner(computer_choice, user_choice):
        '''
        Returns the winner of Rock,Paper,Scissors from the Computer and User choices
        '''
        if computer_choice == "rock" and user_choice == "paper":
            return("user wins")
        elif computer_choice == "rock" and user_choice == "scissors":
            return("computer wins")
        elif computer_choice == "paper" and user_choice == "scissors":
            return("user wins")
        elif computer_choice == "paper" and user_choice == "rock":
            return("computer wins")
        elif computer_choice == "scissors" and user_choice == "rock":
            return("user wins")
        elif computer_choice == "scissors" and user_choice == "paper":
            return("computer wins")
        elif computer_choice == user_choice:
            return("tie")
        else:
            return("Error in user input")

    computer_choice = get_computer_choice()
    user_choice = get_prediction()

    while user_choice == "nothing":
        user_choice = get_prediction()

    print("User guessed:", user_choice)
    print("computer guessed:", computer_choice)
    return(get_winner(computer_choice, user_choice))    
                
        

if __name__ == '__main__':
    print(play())