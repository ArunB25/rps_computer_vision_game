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
