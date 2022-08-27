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

""" ```