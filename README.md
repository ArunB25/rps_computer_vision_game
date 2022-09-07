# Computer_vision
Rock Paper scissors game using computer vision

Image project model created on teachable machine, to regonise 4 different classes, rock,paper,scissors and nothing. Model is on the keras_model.h5 file
code snippets to use model (from Teachable Machine)

```python """
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
image = Image.open('<IMAGE_PATH>')
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
print(prediction)
"""
```
Manual version of Rock-Paper-Scissors created, which uses python random module to make computers decision and takes user input.

```python """
import random

def play():
    def get_computer_choice():
        '''
        Returns random value of rock, paper or scissors
        '''
        options = ["rock","paper","scissors"]
        return random.choice(options)

    def get_user_choice():
        '''
        Gets user input for Rock,Paper or scissors
        '''
        options = ["rock","paper","scissors"]
        user_choice = input("Enter: Rock, Paper or Scissors | ")

        while True:
            if user_choice.lower() in options:
                return user_choice.lower()
            else:
                user_choice = input("Invalid input Enter: Rock, Paper or Scissors | ")

    def get_winner(computer_choice, user_choice):
        if computer_choice == "rock" and user_choice == "paper":
            return("user wins")
        elif computer_choice == "rock" and user_choice == "scissors":
            return("computer wins")
        elif computer_choice == "paper" and user_choice == "scissors":
            return("user wins")
        elif computer_choice == "paper" and user_choice == "rock":
            return("computer wins")
        elif computer_choice == "scissor" and user_choice == "rock":
            return("user wins")
        elif computer_choice == "scissor" and user_choice == "paper":
            return("computer wins")
        else:
            return("tie")
    
    computer_choice = get_computer_choice()
    user_choice = get_user_choice()
    return(get_winner(computer_choice, user_choice))    


if __name__ == '__main__':

    print(play())

""" 
```
Using the cv2 libray the images from the camera on the computer can be used in conjunction with the keras_model.h5 from teachable machine, to make predictions on what sign (rock, paper scissors or nothing) is in the camera image. the predictions are outputted in a list which contains the probability for each sign.

```python """
import cv2
from keras.models import load_model
import numpy as np
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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
"""
```
Building upon the while loop and inserting modules from the manual rock paper scissors game the user predictions come prome the computers camera. The camera images are displayed in a pop up window, text can be written over the images so that the game can be played in the pop up window. 

``` python """
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
    
"""
```



