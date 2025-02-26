import numpy as np
import random 

class TicTacToe():
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.player0 = -1
        self.player1 = 1
        self.EndFlag = False
        self.winner = 0
        self.history = list()




    def move(self, player,seed):

        # Get empty cells not being played yet
        empty_cells = np.argwhere(self.board == 0)

        # First, check if any cell is still empty 
        if len(empty_cells) == 0:
            # It's a draw
            print("It's a draw MOVE")
            self.EndFlag = True
            self.history.append(','.join(str(int(num)) for row in self.board for num in row))
        # elif len(empty_cells) == 1:
        #     self.history.append(','.join(str(int(num)) for row in self.board for num in row))

        #     # Pick random index
        #     random.seed(seed)
        #     random_index = random.choices(empty_cells)
        #     self.board[random_index[0][0]][random_index[0][1]] = player

        #     if self.checkState(player) ==

        else:
            self.history.append(','.join(str(int(num)) for row in self.board for num in row))

            # Pick random index
            random.seed(seed)
            random_index = random.choices(empty_cells)
            self.board[random_index[0][0]][random_index[0][1]] = player

            # self.placement(player, random_index[0][0], random_index[0][1])

        # When only 5 cells left, first player is able to win
        if len(empty_cells) <= 5: 
           self.winner = self.checkState(player)


    def placement(self, player, new_state):
        #self.board[row][column] = player
        self.board = new_state

        empty_cells = np.argwhere(self.board == 0)
        if len(empty_cells) <= 5: 
           self.winner = self.checkState(player)

        if len(empty_cells) == 0:
            print("It's a draw PLACEMENT")
            self.EndFlag = True

        print("PLACEMENT ", self.board)
        
        
        

    def checkState(self, player):       
        # 1. Check: Sum up all horizontally
        for i in range(3):
            if sum(self.board[i, :]) in [3, -3]:
                print(f"Player {player} won horizontally")
                self.EndFlag = True
                return 0 if player == 1 else 1
            elif sum(self.board[:, i]) in [3, -3]:
                print(f"Player {player} won vertically")
                self.EndFlag = True
                return 0 if player == 1 else 1
        
        if sum(self.board.diagonal()) in [3, -3]:
            print(f"Player {player} won diagonally left top to bottom right")
            self.EndFlag = True
            return 0 if player == 1 else 1
        elif sum(np.fliplr(self.board).diagonal()) in [3, -3]:
            print(f"Player {player} won diagonally right top to bottom left")
            self.EndFlag = True
            return 0 if player == 1 else 1
        
        return 0.5

        

    # def play(self):

    #     for i in range(9):
            
    #         if self.EndFlag:
    #             print("End Flag ", self.board)
    #             self.history.append(','.join(str(int(num)) for row in self.board for num in row))
    #             break
    #         else:   
    #             if i % 2 == 0:
    #                 self.move(self.player0)
    #             else:
    #                 self.move(self.player1)
    #     print(self.board)
            
    #     return self.history, self.winner


# if __name__ == "__main__":
#     game = TicTacToe()

#     print(game.play())
