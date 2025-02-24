import numpy as np
import random 

class TicTacToe():
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.player0 = -1
        self.player1 = 1
        self.EndFlag = False



    def move(self, player):

        # Get empty cells not being played yet
        empty_cells = np.argwhere(self.board == 0)

        # First, check if any cell is still empty 
        if len(empty_cells) == 0:
            # It's a draw
            print("It's a draw")
            self.EndFlag = True
        else:
            # Pick random index
            random_index = random.choices(empty_cells)

            if not game.board[random_index[0][0]][random_index[0][1]]:
                # This condition is true, when the spot of the board has been placed yet
                # Set the number of the player inside the board spot i.e. matrix cell
                game.board[random_index[0][0]][random_index[0][1]] = player

        # When only 5 cells left, first player is able to win
        if len(empty_cells) <= 5: 
           self.checkState(player)

    def checkState(self, player):       
        # 1. Check: Sum up all horizontally
        for i in range(3):
            if sum(self.board[i, :]) in [3, -3]:
                print(f"Player {player} won horizontally")
                self.EndFlag = True
            elif sum(self.board[:, i]) in [3, -3]:
                print(f"Player {player} won vertically")
                self.EndFlag = True
        
        if sum(self.board.diagonal()) in [3, -3]:
            print(f"Player {player} won diagonally left top to bottom right")
            self.EndFlag = True
        elif sum(np.fliplr(self.board).diagonal()) in [3, -3]:
            print(f"Player {player} won diagonally right top to bottom left")
            self.EndFlag = True

        

    def play(self):

        for i in range(9):
            
            if self.EndFlag:
                break
            else:   
                if i % 2 == 0:
                    self.move(self.player0)
                else:
                    self.move(self.player1)
            print(self.board)
            


if __name__ == "__main__":
    game = TicTacToe()

    game.play()
