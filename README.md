# Computer_vision
Rock Paper scissors game using computer vision

Image project model created on teachable machine, to regonise 4 different classes, rock,paper,scissors and nothing. Model is on the keras_model.h5 file
code snippets to use model (from Teachable Machine)

![image](https://user-images.githubusercontent.com/111798251/188868460-fcf4bc71-dafe-41e6-841c-3cb1300fcbaa.png)

Manual version of Rock-Paper-Scissors was created which is played in the terminal. It uyses python random module to make computers decision and takes a typed user input.

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
The RPS_template.py uses the cv2 libray to get images from the camera on the computer, in conjunction with the keras_model.h5 from teachable machine, to make predictions on what sign (rock, paper scissors or nothing) is in the camera image. the predictions are outputted in a list which contains the probability for each sign.

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
![image](https://user-images.githubusercontent.com/111798251/188867903-a1103e6a-1fa7-40f5-b7b7-4fd8c596dd6a.png)    


Building upon the while loop and inserting modules from the manual rock paper scissors game the user predictions come prome the computers camera. The camera images are displayed in a pop up window, text can be written over the images so that the game can be played in the pop up window. The game has different states so that the while loops keeps circulating to update the images and have a high frame rate for the video.

![image](https://user-images.githubusercontent.com/111798251/188867012-417790f1-f141-4e40-8b87-f61ed9656a2b.png)
![image](https://user-images.githubusercontent.com/111798251/188867268-8c4d615e-fecd-4f1d-983c-f96c7c18009e.png)




