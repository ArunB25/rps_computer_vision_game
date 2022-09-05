import imp
from time import time
from tkinter.tix import Tree
import cv2
from keras.models import load_model
import numpy as np
import time
import random



def play():

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

    #Game Initialisation
    model = load_model('keras_model.h5')
    cap = cv2.VideoCapture(0)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    user_wins = 0
    computer_wins = 0
    game_stage = 5
    prompt_string = "Start Game Press S"
    quit_print = "Press Q to quit"
    winner_string = ""
    user_guess_string = ""
    computer_guess_string = ""

    while True: 
        ret, frame = cap.read()
        frame = cv2.flip(frame,1) # mirror image
        resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
        data[0] = normalized_image
        prediction = model.predict(data)
        
        rps_prediction = np.argmax(prediction)

        if rps_prediction == 0:
            current_guess = "rock"  
        elif rps_prediction == 1:
            current_guess = "paper"
        elif rps_prediction == 2:
            current_guess = "scissors"
        elif rps_prediction == 3:
            current_guess = "nothing"
        font = cv2.FONT_HERSHEY_SIMPLEX
        current_guess_string = "Current guess: {}".format(current_guess)
        cv2.putText(frame,current_guess_string,(30,60), font, 0.6, (10, 10, 200), 1, cv2.LINE_AA)
        cv2.putText(frame,prompt_string,(30,40), font, 1, (10, 10, 10), 4, cv2.LINE_AA)
        cv2.putText(frame,quit_print,(535,10), font, 0.4, (10, 10, 10), 1, cv2.LINE_AA)
        cv2.putText(frame,winner_string,(30,470), font, 0.8, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.putText(frame,user_guess_string,(30,420), font, 0.7, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.putText(frame,computer_guess_string,(30,445), font, 0.7, (10, 10, 10), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):   # Press q to close the window
            break

       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if user_wins >= 3 or computer_wins >= 3:
            game_stage = 7
   
        if game_stage == 1:
            start_time = time.time()
            current_time = time.time()
            prompt_string = "Show hand in 3"
            game_stage = 2
        elif game_stage == 2:
            current_time = time.time()
            if current_time >= start_time +1 and current_time <= start_time +2 :
                prompt_string = "Show hand in 2"    
            elif current_time >= start_time + 2 and current_time <= start_time +3:
                prompt_string = "Show hand in 1"
            elif current_time >= start_time + 3:
                game_stage = 3    
        elif game_stage == 3:
            prompt_string = "Show hand"
            computer_guess = get_computer_choice()
            user_guess = current_guess
            if user_guess != "nothing":
                game_stage = 4   
        elif game_stage == 4:
            winner = get_winner(computer_guess, user_guess)
            user_guess_string = "User guessed: {}".format(user_guess)
            computer_guess_string = "Computer guessed: {}".format(computer_guess)
            if winner == "user":
                user_wins += 1
                winner_string = "User won! Score User {} | Computer {} ".format(user_wins, computer_wins)
                game_stage = 5
            elif winner == "computer":
                computer_wins += 1
                winner_string = "Computer won! Score User {} | Computer {} ".format(user_wins, computer_wins)
                game_stage = 5
            elif winner == "tie":
                winner_string = "Tie! Score User {} | Computer {} ".format(user_wins, computer_wins)
                game_stage = 5
        elif game_stage == 5:
            start_time = time.time()
            current_time = time.time()
            prompt_string = "Next round in 5"
            game_stage = 6
        elif game_stage == 6:
            current_time = time.time()
            if current_time >= start_time +1 and current_time <= start_time +2 :
                prompt_string = "Next round in 4"    
            elif current_time >= start_time + 2 and current_time <= start_time +3:
                prompt_string = "Next round in 3"
            elif current_time >= start_time + 3 and current_time <= start_time +4:
                prompt_string = "Next round in 2"
            elif current_time >= start_time + 4 and current_time <= start_time +5:
                prompt_string = "Next round in 1"
                user_guess_string = ""
                computer_guess_string = ""
            elif current_time >= start_time + 5:
                game_stage = 1 
        elif game_stage == 7:
            start_time = time.time()
            current_time = time.time()
            if user_wins == 3:
                winner_string = "User Won Overall!"
            elif computer_wins == 3:
                winner_string = "Computer Won Overall"
            prompt_string = "Game Closes in 5"
            user_wins = 0
            computer_wins = 0
            game_stage = 8
        elif game_stage == 8:
            current_time = time.time()
            if current_time >= start_time +1 and current_time <= start_time +2 :
                prompt_string = "Game Closes in 4"  
                user_guess_string = ""
                computer_guess_string = ""  
            elif current_time >= start_time + 2 and current_time <= start_time +3:
                prompt_string = "Game Closes in 3"
            elif current_time >= start_time + 3 and current_time <= start_time +4:
                prompt_string = "Game Closes in 2"
            elif current_time >= start_time + 4 and current_time <= start_time +5:
                prompt_string = "Game Closes in 1"
            elif current_time >= start_time + 5:
                break
    

         
    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print(play())
   
    