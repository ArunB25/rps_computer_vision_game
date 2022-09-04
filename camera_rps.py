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
            return("user")
        elif computer_choice == "rock" and user_choice == "scissors":
            return("computer")
        elif computer_choice == "paper" and user_choice == "scissors":
            return("user")
        elif computer_choice == "paper" and user_choice == "rock":
            return("computer")
        elif computer_choice == "scissors" and user_choice == "rock":
            return("user")
        elif computer_choice == "scissors" and user_choice == "paper":
            return("computer")
        elif computer_choice == user_choice:
            return("tie")
        else:
            return("Error in user input")
    
    def countdown():
        start_time = time.time()
        current_time = time.time()
        print("3")
        while start_time + 3 != current_time:
            if current_time == start_time +1:
                print("2")
            elif current_time == start_time + 2:
                print("1")
            current_time = time.time()
        print("show hand")
    
    user_wins = 0
    computer_wins = 0

    while user_wins < 3 and computer_wins < 3:
        computer_choice = get_computer_choice()
        countdown()

        user_choice = get_prediction()
        while user_choice == "nothing":
            user_choice = get_prediction()

        print("User guessed:", user_choice)
        print("computer guessed:", computer_choice)
        winner = get_winner(computer_choice, user_choice)

        if winner == "user":
            user_wins += 1
            print("User won! Score User ", user_wins , "| Computer ", computer_wins)
        elif winner == "computer":
            computer_wins += 1
            print("Computer won! Score User ", user_wins , "| Computer ", computer_wins)
        elif winner == "tie":
            print("That was a Tie Score User ", user_wins , "| Computer ", computer_wins)

    if user_wins == 3:
        return("User Won Overall")
    elif computer_wins == 3:
        return("Computer Won Overall")
    
                
        

if __name__ == '__main__':
    print(play())