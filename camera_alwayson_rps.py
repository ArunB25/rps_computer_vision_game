import imp
from operator import truediv
from time import time
from tkinter.tix import Tree
import cv2
from keras.models import load_model
import numpy as np
import time
import random

class play_rps:

    def __init__(self):
        '''
        setup the initial functions for the game
        '''
        self.model = load_model('keras_model.h5')
        self.cap = cv2.VideoCapture(0)
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.user_wins = 0
        self.computer_wins = 0
        self.game_stage = 5
        self.prompt_string = "Game Started"
        self.quit_print = "Press Q to quit"
        self.winner_string = ""
        self.user_choice_string = ""
        self.computer_choice_string = ""

    def get_computer_choice(self):
                '''
                Returns random value of rock, paper or scissors
                '''
                options = ["rock","paper","scissors"]
                self.computer_choice = random.choice(options)

    def get_winner(self):
        '''
        Gets the winner of Rock,Paper,Scissors from the Computer and User choices and updates overlay strings and winner counts
        '''
        if self.computer_choice == "rock" and self.user_choice == "paper":
            winner = "user"
        elif self.computer_choice == "paper" and self.user_choice == "scissors":
             winner = "user"
        elif self.computer_choice == "scissors" and self.user_choice == "rock":
            winner = "user"
        elif self.computer_choice == self.user_choice:
            winner = "tie"
        else:
            winner = "computer"

        self.user_choice_string = "User chose: {}".format(self.user_choice)
        self.computer_choice_string = "Computer chose: {}".format(self.computer_choice)
        if winner == "user":
            self.user_wins += 1
            self.winner_string = "User won! Score User {} | Computer {} ".format(self.user_wins, self.computer_wins)
            self.game_stage = 5
        elif winner == "computer":
            self.computer_wins += 1
            self.winner_string = "Computer won! Score User {} | Computer {} ".format(self.user_wins, self.computer_wins)
            self.game_stage = 5
        elif winner == "tie":
            self.winner_string = "Tie! Score User {} | Computer {} ".format(self.user_wins, self.computer_wins)

    def video_prediction(self):
        '''
        get image from camera, formats image, then passes it through the ML model to get predictions and selects highest probability
        
        '''
        ret, frame = self.cap.read()
        self.frame = cv2.flip(frame,1) # mirror image
        resized_frame = cv2.resize(self.frame, (224, 224), interpolation = cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
        self.data[0] = normalized_image
        prediction = self.model.predict(self.data)
        rps_prediction = np.argmax(prediction)

        if rps_prediction == 0:
            self.current_choice = "rock"  
        elif rps_prediction == 1:
            self.current_choice = "paper"
        elif rps_prediction == 2:
            self.current_choice = "scissors"
        elif rps_prediction == 3:
            self.current_choice = "nothing"

    def video_overlay(self):
        """
        overlays text on to the current frame and displays it
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        current_choice_string = "Current choice: {}".format(self.current_choice)
        cv2.putText(self.frame,current_choice_string,(30,60), font, 0.6, (10, 10, 200), 1, cv2.LINE_AA)
        cv2.putText(self.frame,self.prompt_string,(30,40), font, 1, (10, 10, 10), 4, cv2.LINE_AA)
        cv2.putText(self.frame,self.quit_print,(535,10), font, 0.4, (10, 10, 10), 1, cv2.LINE_AA)
        cv2.putText(self.frame,self.winner_string,(30,470), font, 0.8, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.putText(self.frame,self.user_choice_string,(30,420), font, 0.7, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.putText(self.frame,self.computer_choice_string,(30,445), font, 0.7, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.imshow('frame', self.frame)

    def showhand_countdown(self):
        """
        countdown from 3 whilst updating video frames and overlays
        """
        start_time = time.time()
        current_time = time.time()
        countdown_stage = 1 #countdown stage only allows if statement that acts between times to occur ones, to prevent unecassary repetition of if statements
        self.prompt_string = "Show hand in 3"
        
        while True:
            current_time = time.time()
            self.video_prediction()
            self.video_overlay()

            if current_time >= start_time +1 and current_time <= start_time +2 and countdown_stage != 2:
                    self.prompt_string = "Show hand in 2"    
                    countdown_stage = 2
            elif current_time >= start_time + 2 and current_time <= start_time +3 and countdown_stage != 3:
                    self.prompt_string = "Show hand in 1"
                    countdown_stage = 3
            elif current_time >= start_time + 3:
                    break   
            
        self.prompt_string = "Show hand"
    
    def nextround_counter(self):
        """
        countdown from 5 whilst updating video frames and overlays
        """
        start_time = time.time()
        current_time = time.time()
        countdown_stage = 1
        self.prompt_string = "Next round in 5"

        while True:
            current_time = time.time()
            self.video_prediction()
            self.video_overlay()

            if current_time >= start_time +1 and current_time <= start_time +2 and countdown_stage != 2:
                self.prompt_string = "Next round in 4"
                countdown_stage = 2    
            elif current_time >= start_time + 2 and current_time <= start_time +3 and countdown_stage != 3:
                self.prompt_string = "Next round in 3"
                countdown_stage = 3
            elif current_time >= start_time + 3 and current_time <= start_time +4 and countdown_stage != 4:
                self.prompt_string = "Next round in 2"
                countdown_stage = 4
            elif current_time >= start_time + 4 and current_time <= start_time +5 and countdown_stage != 5:
                self.prompt_string = "Next round in 1"
                self.user_choice_string = ""
                self.computer_choice_string = ""
                countdown_stage = 5
            elif current_time >= start_time + 5:
                break

    def endgame_countdown(self):
        start_time = time.time()
        current_time = time.time()
        countdown_stage = 1
        if self.user_wins == 3:
            self.winner_string = "User Won Overall!"
        elif self.computer_wins == 3:
            self.winner_string = "Computer Won Overall"
        self.prompt_string = "Game Closes in 5"

        while True:
            current_time = time.time()
            self.video_prediction()
            self.video_overlay()

            if current_time >= start_time +1 and current_time <= start_time +2 and countdown_stage != 2:
                self.prompt_string = "Game Closes in 4"  
                self.user_choice_string = ""
                self.computer_choice_string = ""
                countdown_stage = 2  
            elif current_time >= start_time + 2 and current_time <= start_time +3 and countdown_stage != 3:
                self.prompt_string = "Game Closes in 3"
                countdown_stage = 3
            elif current_time >= start_time + 3 and current_time <= start_time +4 and countdown_stage != 4:
                self.prompt_string = "Game Closes in 2"
                countdown_stage = 4
            elif current_time >= start_time + 4 and current_time <= start_time +5 and countdown_stage != 5:
                self.prompt_string = "Game Closes in 1"
                countdown_stage = 5
            elif current_time >= start_time + 5:
                break


    def play(self):

        while True: 
            
            #self.video_prediction()
            #self.video_overlay()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):   # Press q to close the window
                break
    
            self.showhand_countdown()
            self.get_computer_choice()
            self.user_choice = self.current_choice
            
            #while self.user_choice != "nothing":
            #    self.video_prediction()
            #    self.video_overlay()
            #    self.user_choice = self.current_choice  

            self.get_winner()
            self.nextround_counter()

            if self.user_wins >= 3 or self.computer_wins >= 3:
                self.endgame_countdown
                # After the loop release the cap object
                self.cap.release()
                # Destroy all the windows
                cv2.destroyAllWindows()
                break
            



if __name__ == '__main__':
    mygame = play_rps()
    mygame.play()
   
    