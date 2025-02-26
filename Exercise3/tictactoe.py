import numpy as np
import random 

class TicTacToeAgent():
    def __init__(self, exploration_rate):
        self.board = np.zeros((3, 3))
        self.player0 = -1
        self.player1 = 1
        self.EndFlag = False
        self.winner = 0
        self.history = list()
        self.board_states={}
        self.alpha=0.5
        self.exploration_rate = exploration_rate

    def move(self, player, seed):

        # Get empty cells not being played yet
        empty_cells = np.argwhere(self.board == 0)

        # First, check if any cell is still empty 
        if len(empty_cells) == 0:
            # It's a draw
            print("It's a (potential) draw")
            self.EndFlag = True
            self.history.append(','.join(str(int(num)) for row in self.board for num in row))
        else:
            self.history.append(','.join(str(int(num)) for row in self.board for num in row))

            # Pick random index
            random.seed(seed)
            random_index = random.choices(empty_cells)
            self.board[random_index[0][0]][random_index[0][1]] = player

        # When only 5 cells left, first player is able to win
        if len(empty_cells) < 5: 
           self.winner = self.checkState(player)
        
        if len(empty_cells) == 0:
            print("It's a (potential) draw")
            self.EndFlag = True


    def manual_move(self, player, user_input_row, user_input_col):

        self.history.append(','.join(str(int(num)) for row in self.board for num in row))

        # only place user marker when input position is valid (matrix empty at that position)
        while (self.board[user_input_row][user_input_col] != 0):
            print("Input invalid. Please choose a valid position for your marker.")
            user_input_row = int(input("Enter row number [0, 1, 2]: "))
            user_input_col = int(input("Enter column number [0, 1, 2]: "))
        
        self.board[user_input_row][user_input_col] = player

        empty_cells = np.argwhere(self.board == 0)

        # check if there's a winner
        if len(empty_cells) < 5: 
           self.winner = self.checkState(player)

        # check if there's a draw
        # EndFlag = True twice if last move was winning move
        if len(empty_cells) == 0:
            print("It's a (potential) draw")
            self.EndFlag = True



    def placement(self, player, new_state):
        
        self.history.append(','.join(str(int(num)) for row in self.board for num in row))
        
        self.board = new_state

        empty_cells = np.argwhere(self.board == 0)
        if len(empty_cells) < 5: 
           self.winner = self.checkState(player)

        # EndFlag = True twice if last move was winning move
        if len(empty_cells) == 0:
            print("It's a (potential) draw")
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
        
        return 0.2
    

    def choose_move(self, current_state, seed):
        np.random.seed(seed)
        if np.random.uniform(low=0, high=1) <= self.exploration_rate:
            # random move
            self.move(-1, seed)
            print("RANDOM UNIFORM ")
            print(self.board)
            return np.zeros((3,3))

        else:
            # greedy move according to best value

            best_value = -1000000
            best_state = ""
            print("CHECK ",self.board_states.get(current_state))
            # make random move if the current state hasn't been encountered yet
            if self.board_states.get(current_state) is None:
                print("first move")
                self.move(-1, seed)
                return np.zeros((3,3))
            else:
                for next_state in self.board_states[current_state]["next_states"]:
                    if (self.board_states[next_state]["value"] > best_value):
                        best_state = next_state
                        best_value = self.board_states[next_state]["value"]

            matrix_best_state = []

            best_state = best_state.split(",")

            for idx, letter in enumerate(best_state):
                matrix_best_state.append(int(letter))

            matrix_best_state = np.array(matrix_best_state).reshape(3, 3)
            return matrix_best_state
        


    def calculate_state_values(self):


        for index, state in enumerate(reversed(self.history)):
            print("State", state)
            if index==0:
                self.board_states[state]={'value': self.winner,'next_states': set()}
                print("debug", self.board_states)
                old_state=state
                continue

            if self.board_states.get(state) is None:
                self.board_states[state] = {'value': 0,'next_states': set()}
            print("Value beginning", self.board_states[state]['value'])

            self.board_states[state]['value'] = (self.board_states[state]['value'] + self.alpha * (self.board_states[old_state]['value'] - self.board_states[state]['value']))
            self.board_states[state]['next_states'].add(old_state)

            print("Value afterwards", self.board_states[state]['value'])
            old_state=state

        # Reset history for every game
        self.history = list()



        
        
    def play(self,seed):
        for i in range(10):
            if self.EndFlag:
                self.history.append(','.join(str(int(num)) for row in self.board for num in row))
                print("HISTORY PLAY ", self.history)
                print("BOARD STATES ", self.board_states)
                break
            else:   
                if i % 2 == 0:
                    print("AGENT IS MOVING")
                    current_state = ','.join(str(int(num)) for row in self.board for num in row)
                    print("CURRENT STATE: ")
                    print(current_state)
                    # print("CURRENT BOARD: ")
                    # print(self.board)
                    new_state = self.choose_move(current_state, seed)
                    print("Agent BOARD after move: ")
                    print(self.board)
                    if np.all(new_state == 0):
                        print("NP.ALL")
                        continue
                    print("NEW STATE (AFTER MOVE): ")
                    print(new_state)
                    self.placement(-1, new_state)
                    
                else:
                    # print("PLAYER RANDOM IS MOVING")
                    # self.move(1,seed)
                    # print(self.board)

                    print("MANUAL USER IS MOVING")
                    print("Current board")
                    print(self.board)
                    user_input_row = int(input("Enter row number [0, 1, 2]: "))
                    user_input_column = int(input("Enter column number [0, 1, 2]: "))
                    self.manual_move(1, user_input_row, user_input_column)
                    print(self.board)
                    

        # Reset all game instances
        self.EndFlag = False
        self.board = np.zeros((3,3))
        self.calculate_state_values()
 
        
        

if __name__ == "__main__":
    RANDOM_SEED = 63
    agent = TicTacToeAgent(exploration_rate=0.2)

    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    random_seed_list=random.sample(range(1,10000), 1)


    for seed in random_seed_list:
        agent.play(seed)

    with open("output.txt", "a") as f:
        print(agent.board_states, file=f)
  

