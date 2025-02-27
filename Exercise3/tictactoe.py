import numpy as np
import random 

DRAW_VALUE_FIRST_PLAYER = 0.1
DRAW_VALUE_SECOND_PLAYER = 0.6
VALUE_THRESHOLD = 0.001
ALPHA = 0.5

class TicTacToeAgent():
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.player0 = -1
        self.player1 = 1
        self.EndFlag = False
        self.winner = 0
        self.history = list()
        self.board_states1={}
        self.board_states2={}
        self.winners_list=list()

    def random_move(self, player, seed):

        # Get empty cells not being played yet
        empty_cells = np.argwhere(self.board == 0)

        # Append the current game state to our game history
        self.history.append(','.join(str(int(num)) for row in self.board for num in row))

        # Pick random index
        random.seed(seed)
        print("random_move seed", seed)
        random_index = random.choices(empty_cells)
        self.board[random_index[0][0]][random_index[0][1]] = player

        # When only 5 cells are left, first player is able to win, so we check for it
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
        
        empty_cells = np.argwhere(self.board == 0)
        if len(empty_cells) == 0:
            self.EndFlag = True

        return 0.5
  

    def explore_move(self, player_name, seed, exploration_rate):
        np.random.seed(seed)
        print("explore move seed", seed)
        random_uniform = np.random.uniform(low=0, high=1)
        print("UNIFORM", random_uniform, exploration_rate)
        if random_uniform <= exploration_rate:
            # random move
            if "1" in player_name:
                self.random_move(-1, seed)
            else:
                self.random_move(1, seed)
            print("RANDOM UNIFORM ")
            #print(self.board)
            return np.zeros((3,3))

        else:


            if "1" in player_name: 
                return self.choose_greedy_move(self.board_states1, -1, seed)
            else:
                return self.choose_greedy_move(self.board_states2, 1, seed)

    def rotate_current_state_for_placing(self, board_states):
        print("ORIGINAL BOARD ", self.board)

        beginning_state = ','.join(str(int(num)) for row in self.board for num in row)

        # Check if current state has been played before by rotating it
        if board_states.get(beginning_state) is not None:
            return 0, beginning_state
        for i in range(1,4):
            print("ROTATE BOARD ", np.rot90(self.board,i))
            rotated_state = ','.join(str(int(num)) for row in np.rot90(self.board,i) for num in row)
            if board_states.get(rotated_state) is not None:
                print("Found a state by rotating", i, " times")
                return i, rotated_state
        
        # If nothing passes, this state has never been played before and needs to be saved
        return -1, beginning_state
        

            
    def choose_greedy_move(self, board_states, player_score, seed):
            
            # greedy move according to best value
            best_value = -1
            best_state = ""

            
            number_of_rotations, current_state = self.rotate_current_state_for_placing(board_states)

            print("CHECK ",board_states.get(current_state))
            # make random move if the current state hasn't been encountered yet
            if board_states.get(current_state) is None and number_of_rotations == -1:
                print("first move")
                self.random_move(player_score, seed)
                return np.zeros((3,3))
            else:
                for next_state in board_states[current_state]["next_states"]:
                    if (board_states[next_state]["value"] > best_value):
                        best_state = next_state
                        best_value = board_states[next_state]["value"]
                
                # Set threshold to make sure, we do not repeat playing loosing games
                if best_value < VALUE_THRESHOLD:
                   self.random_move(player_score, seed)
                   return np.zeros((3,3))

            matrix_best_state = []

            best_state = best_state.split(",")

            for letter in best_state:
                matrix_best_state.append(int(letter))

            matrix_best_state = np.array(matrix_best_state).reshape(3, 3)

            # Rotate matrix back i.e. take absolute value of - 4 to have a 360° rotation 
            matrix_best_state_original_rotation = np.rot90(matrix_best_state, abs(number_of_rotations-4))
            return matrix_best_state_original_rotation
        

    def rotate_state_for_history(self, state, number_of_rotations):
        
        restored_matrix = []

        splitted_state = state.split(",")

        for letter in splitted_state:
            restored_matrix.append(int(letter))

        restored_matrix = np.array(restored_matrix).reshape(3, 3)
        # print("RESTORED MATRIX")
        # print(restored_matrix)
        # print("ROTATED MATRIX")
        # print(np.rot90(restored_matrix,number_of_rotations))

        rotated_state = ','.join(str(int(num)) for row in np.rot90(restored_matrix,number_of_rotations) for num in row)

        return rotated_state
    
    def check_necessary_rotations(self, board_states):
        for index, state in enumerate(self.history):
            if index == 0:
                continue

            if board_states.get(state) is not None:
                return 0

            restored_matrix = []

            splitted_state = state.split(",")

            for letter in splitted_state:
                restored_matrix.append(int(letter))

            restored_matrix = np.array(restored_matrix).reshape(3, 3)

            # Check if current state has been played before by rotating it
            for i in range(1,4):
                rotated_state = ','.join(str(int(num)) for row in np.rot90(restored_matrix,i) for num in row)
                if board_states.get(rotated_state) is not None:
                    return i
                
        # If no state has ever been played return 0 
        return 0
        

    def calculate_state_values(self, player_name):
        is_player1 = "1" in player_name
        print("is_player1 ", is_player1)
        board_states = self.board_states1 if is_player1 else self.board_states2
        draw_value = DRAW_VALUE_FIRST_PLAYER if is_player1 else DRAW_VALUE_SECOND_PLAYER

        print("HISTORY", self.history)

        number_of_rotations = self.check_necessary_rotations(board_states)
        
        for index, state in enumerate(reversed(self.history)):

            # Only rotate the game states if
            if number_of_rotations != 0:
                state = self.rotate_state_for_history(state, number_of_rotations)         
            
            if board_states.get(state) is None:
                board_states[state] = {'value': 0.5, 'next_states': set()}
            
            if index == 0:
                board_states[state]['value'] = draw_value if self.winner == 0.5 else (self.winner if is_player1 else (0 if self.winner == 1 else 1))
            else:
                board_states[state]['value'] += ALPHA * (board_states[old_state]['value'] - board_states[state]['value'])
                board_states[state]['next_states'].add(old_state)
            
            
            old_state = state
            




    def learning_agent_move(self, player_name, exploration_rate, seed):

        # TODO: Do you know this state? 
        # Yes, then look at the next best states
        # No, then rotate the state three times and check if you have seen one of these states before
        # Take this rotated state and the degrees of rotation as an input
        # current_state = ','.join(str(int(num)) for row in self.board for num in row)

        new_state = self.explore_move(player_name, seed, exploration_rate)
        # Catch case of all zero matrix meaning a random move had to be drawn
        
        if np.all(new_state == 0):
            return 0
        
        if "1" in player_name:
            self.placement(-1, new_state)
        else: 
            self.placement(1, new_state)
        

    def play(self,seed, player1, player2, exploration_rate):
        for i in range(10):
            if self.EndFlag:
                self.history.append(','.join(str(int(num)) for row in self.board for num in row))
                break
            else:   
                if i % 2 == 0:
                    if player1 == "random_agent1":
                        print("RANDOM AGENT 1 IS MOVING")
                        self.random_move(-1, seed*(i+1))
                        print(self.board)
                    elif player1 == "user":
                        print("MANUAL USER IS MOVING")
                        print("Current board")
                        print(self.board)
                        user_input_row = int(input("Enter row number [0, 1, 2]: "))
                        user_input_column = int(input("Enter column number [0, 1, 2]: "))
                        self.manual_move(-1, user_input_row, user_input_column)
                        print(self.board)
                    else:
                        print("LEARNING AGENT 1 IS MOVING")
                        self.learning_agent_move(player1, exploration_rate, seed*(i+1))
                        print(self.board)
                else:                 
                    if player2 == "random_agent2":
                        print("RANDOM AGENT 2 IS MOVING")
                        self.random_move(1, seed*(i+1))
                        print(self.board)
                    elif player2 == "user":
                        print("MANUAL USER IS MOVING")
                        print("Current board")
                        print(self.board)
                        user_input_row = int(input("Enter row number [0, 1, 2]: "))
                        user_input_column = int(input("Enter column number [0, 1, 2]: "))
                        self.manual_move(1, user_input_row, user_input_column)
                        print(self.board)
                    else:
                        print("LEARNING AGENT 2 IS MOVING")
                        self.learning_agent_move(player2, exploration_rate, seed*(i+1))
                        print(self.board)
                        


        # Calculate state values for learning agents
        if "learning" in player1:
            print("LEARNING AGENT 1")
            self.calculate_state_values(player1)
        if "learning" in player2:
            print("LEARNING AGENT 2")
            self.calculate_state_values(player2)
 
        # Reset all game instances
        self.winners_list.append(self.winner) 
        self.EndFlag = False
        self.board = np.zeros((3,3))
        self.history = list()


        


if __name__ == "__main__":
    RANDOM_SEED = 67
    agent = TicTacToeAgent()




    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    random_seed_list=random.sample(range(1,1000000), 10000)
    


    for seed in random_seed_list:
        print("SEED",seed)
        
        agent.play(seed, "learning_agent1", "learning_agent2", 0.1)
    


    with open('output_board_states2.json', 'w') as f:
        f.write(str(agent.board_states2))

    with open('output_board_states1.json', 'w') as f:
        f.write(str(agent.board_states1))
    

    with open('winner.txt', 'w') as f:
        f.write(str(agent.winners_list))


    agent.play(51, "user", "learning_agent2", 0)
    agent.play(51, "learning_agent1", "user", 0)

