import numpy as np
import random 

DRAW_VALUE_FIRST_PLAYER = 0.1
DRAW_VALUE_SECOND_PLAYER = 0.6
VALUE_THRESHOLD = 0.001
LEARNING_RATE = 0.5
EXPLORATION_RATE = 0.3
RANDOM_SEED = 67
NUMBER_OF_ITERATIONS = 10000

class TicTacToeAgent():
    """
    A class representing an AI agent for the game of Tic-Tac-Toe.

    Attributes:
        board (np.ndarray): A 3x3 numpy array representing the game board.
        player0 (int): Value representing player 0 (default: -1).
        player1 (int): Value representing player 1 (default: 1).
        EndFlag (bool): A flag indicating whether the game has ended.
        winner (int): The winner of the game (0 if no winner, -1 for player 0, 1 for player 1).
        history (list): A list tracking the history of moves made in the game.
        board_states1 (dict): A dictionary to store board states for player 1.
        board_states2 (dict): A dictionary to store board states for player 2.
        winners_list (list): A list to keep track of winners across multiple games.
    """

    def __init__(self):
        """
        Initializes a new TicTacToeAgent instance.

        The game board is initialized as a 3x3 grid of zeros, representing empty spaces.
        Player markers are set as -1 for player 0 and 1 for player 1.
        Other attributes track the game state, history, and winners.
        """
        self.board = np.zeros((3, 3))
        self.player0 = -1
        self.player1 = 1
        self.EndFlag = False
        self.winner = 0
        self.history = list()
        self.board_states1 = {}
        self.board_states2 = {}
        self.winners_list = list()


    def random_move(self, player, seed):
        """
        Makes a random move for the given player.

        This method selects a random empty cell on the board and places the player's marker there.
        It also updates the game history and checks if a winner exists when fewer than five cells remain.

        Args:
            player (int): The player's identifier (-1 for player 0, 1 for player 1).
            seed (int): The seed value for the random number generator to ensure reproducibility.

        Updates:
            - Adds the current board state to the history.
            - Places the player's move on a randomly chosen empty cell.
            - Checks for a winner when fewer than five cells remain.
            - Sets `EndFlag` to True if the board is full, indicating a draw.
        """

        # Get empty cells not being played yet
        empty_cells = np.argwhere(self.board == 0)

        # Append the current game state to our game history
        self.history.append(','.join(str(int(num)) for row in self.board for num in row))

        # Pick random index
        random.seed(seed)
        random_index = random.choices(empty_cells)
        self.board[random_index[0][0]][random_index[0][1]] = player

        self.check_game_state(player)

    def check_game_state(self, player):
        """
        Checks the current state of the game after a player's move.

        This method evaluates the game board to determine if a player has won or if the game has ended in a draw.
        It is triggered after a move is made and updates the game state accordingly.

        Args:
            player (int): The player's identifier (-1 for player 0, 1 for player 1).

        Updates:
            - Checks for a winner if fewer than five empty cells remain.
            - Sets `winner` if a player wins the game.
            - Marks `EndFlag` as True if the board is full, indicating a draw.
        """

        # Update empty cells after placing the move
        empty_cells = np.argwhere(self.board == 0)

        # When only 5 cells are left, the first player can win, so we check for it
        if len(empty_cells) < 5: 
            print("Entering empty cells < 5", empty_cells)
            self.winner = self.checkState(player)
        
        if len(empty_cells) == 0:
            print("It's a draw")
            self.EndFlag = True

    def checkState(self, player):  
        """
        Evaluates the current board state to determine if a player has won.

        This method checks for winning conditions by summing up values across rows, columns, and diagonals.
        If a player wins, the `EndFlag` is set to True, and the winning player is returned.
        If no winner is found, the method returns 0.5, indicating the game is still ongoing.

        Args:
            player (int): The player's identifier (-1 for player 0, 1 for player 1).

        Returns:
            int: 0 if player 1 (-1) wins, 1 if player 2 (1) wins, or 0.5 if no winner is determined.

        Updates:
            - Checks horizontal, vertical, and diagonal sums for a winning condition.
            - Sets `EndFlag` to True if a win is detected.
        """     
    
        # 1. Check: Sum up all values horizontally
        for i in range(3):
            if sum(self.board[i, :]) in [3, -3]:
                print(f"Player {player} won horizontally")
                self.EndFlag = True
                return 0 if player == 1 else 1
            # 2. Check: Sum up all values vertically
            elif sum(self.board[:, i]) in [3, -3]:
                print(f"Player {player} won vertically")
                self.EndFlag = True
                return 0 if player == 1 else 1

        # 3. Check: Sum up all values in both diagonals
        if sum(self.board.diagonal()) in [3, -3]:
            print(f"Player {player} won diagonally left top to bottom right")
            self.EndFlag = True
            return 0 if player == 1 else 1
        elif sum(np.fliplr(self.board).diagonal()) in [3, -3]:
            print(f"Player {player} won diagonally right top to bottom left")
            self.EndFlag = True
            return 0 if player == 1 else 1

        return 0.5


    def manual_move(self, player, user_input_row, user_input_col):
        """
        Allows a player to manually place a move on the Tic-Tac-Toe board.

        This method takes user input for a move and ensures the chosen position is valid.
        If the input position is already occupied, the user is prompted to enter a new one.
        The move is then recorded on the board, and the game state is updated accordingly.

        Args:
            player (int): The player's identifier (-1 for player 0, 1 for player 1).
            user_input_row (int): The row index (0, 1, or 2) where the player wants to place their marker.
            user_input_col (int): The column index (0, 1, or 2) where the player wants to place their marker.

        Updates:
            - Adds the current board state to the history.
            - Ensures the selected move is placed in a valid, empty cell.
            - Checks if a player has won when fewer than five empty cells remain.
            - Sets `EndFlag` to True if the board is full, indicating a draw.
        """

        self.history.append(','.join(str(int(num)) for row in self.board for num in row))

        # Only place user marker when input position is valid (matrix empty at that position)
        while self.board[user_input_row][user_input_col] != 0:
            print("Input invalid. Please choose a valid position for your marker.")
            user_input_row = int(input("Enter row number [0, 1, 2]: "))
            user_input_col = int(input("Enter column number [0, 1, 2]: "))

        self.board[user_input_row][user_input_col] = player

        self.check_game_state(player)



    def placement(self, player, new_state):
        """
        Updates the game board with a new state for a given player.

        This method replaces the current board state with a new one and updates the game history.
        It also checks if there is a winner when fewer than five empty cells remain.

        Args:
            player (int): The player's identifier (-1 for player 0, 1 for player 1).
            new_state (np.ndarray): A 3x3 numpy array representing the new board state.

        Updates:
            - Adds the previous board state to the history.
            - Replaces the current board with `new_state`.
            - Checks if a player has won when fewer than five empty cells remain.
        """

        self.history.append(','.join(str(int(num)) for row in self.board for num in row))

        self.board = new_state

        self.check_game_state(player)
        

    def rotate_current_state_for_placing(self, board_states):
        """
        Checks if the current board state has been encountered before, considering rotations.

        This method generates the hashed representation of the current board state and checks 
        if it exists in the `board_states` dictionary. If not found, it iterates through 
        90-degree rotations to find a match. If no match is found, it returns an indicator 
        that the state is new.

        Args:
            board_states (dict): A dictionary mapping hashed board states to their values.

        Returns:
            tuple:
                - int: The number of 90-degree rotations (0, 1, 2, or 3) needed to match a known state, 
                    or -1 if the state is new.
                - str: The hashed representation of the matched or new board state.
        """
        
        beginning_state = ','.join(str(int(num)) for row in self.board for num in row)

        # Check if current state has been played before by rotating it
        if board_states.get(beginning_state) is not None:
            return 0, beginning_state
        for i in range(1,4):
            rotated_state = ','.join(str(int(num)) for row in np.rot90(self.board,i) for num in row)
            if board_states.get(rotated_state) is not None:
                return i, rotated_state
        
        # If nothing passes, this state has never been played before and needs to be saved
        return -1, beginning_state
        

            
    def choose_greedy_move(self, board_states, player_score, seed):
        """
        Selects the next move using a greedy policy based on learned state values.

        This method determines the best possible move by choosing the state with the highest 
        value from the set of known next states. If no known states exist, or if the best 
        value is below a predefined threshold, the method falls back to a random move.

        Args:
            board_states (dict): A dictionary mapping hashed board states to their values 
                                and possible next states.
            player_score (int): The score representation of the player (-1 for player 1, 1 for player 2).
            seed (int): A seed value for reproducibility in random move selection.

        Returns:
            np.ndarray: A 3x3 NumPy array representing the chosen board state after the move.
                        Returns a zero matrix if a random move is made.
        """


        # greedy move according to best value
        best_value = -1
        best_state = ""

        
        number_of_rotations, current_state = self.rotate_current_state_for_placing(board_states)

        print("CHECK ",board_states.get(current_state))
        # make random move if the current state hasn't been encountered yet
        if board_states.get(current_state) is None and number_of_rotations == -1:
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

        matrix_best_state = self.restore_matrix_from_hashed_state(best_state)

        # Rotate matrix back i.e. take absolute value of - 4 to have a 360° rotation 
        matrix_best_state_original_rotation = np.rot90(matrix_best_state, abs(number_of_rotations-4))
        return matrix_best_state_original_rotation
        

    def restore_matrix_from_hashed_state(self, state):
        """
        Converts a hashed board state back into a 3x3 NumPy matrix.

        The hashed state is a string representation where board values are stored 
        as comma-separated integers. This method splits the string, converts 
        the values to integers, and reshapes them into a 3x3 matrix.

        Args:
            state (str): The hashed board state as a comma-separated string.

        Returns:
            np.ndarray: A 3x3 NumPy array representing the Tic-Tac-Toe board.
        """

        restored_matrix = []

        splitted_state = state.split(",")

        for letter in splitted_state:
            restored_matrix.append(int(letter))

        restored_matrix = np.array(restored_matrix).reshape(3, 3)   

        return restored_matrix     


    def rotate_state_for_history(self, state, number_of_rotations):
        """
        Rotates a given board state by a specified number of 90-degree increments.

        This method restores the board state from its hashed representation, applies 
        the given number of 90-degree rotations, and then converts it back into a 
        hashed state format.

        Args:
            state (str): The hashed representation of the board state.
            number_of_rotations (int): The number of 90-degree rotations to apply (1, 2, or 3).

        Returns:
            str: The rotated board state in hashed format.
        """

        restored_matrix = self.restore_matrix_from_hashed_state(state)

        rotated_state = ','.join(str(int(num)) for row in np.rot90(restored_matrix,number_of_rotations) for num in row)

        return rotated_state
    

    def check_necessary_rotations(self, board_states):
        """
        Determines the number of 90-degree rotations needed to match a previously seen board state.

        This method checks if any of the states in the game history have been encountered 
        before in the `board_states` dictionary. If a state is found in its original or 
        rotated form, it returns the number of rotations required to match it.

        Args:
            board_states (dict): A dictionary mapping board states to their values.

        Returns:
            int: The number of 90-degree rotations (1, 2, or 3) needed to match a known state, 
                or 0 if no match is found.
        """
        
        for index, state in enumerate(self.history):
            if index == 0:
                continue

            if board_states.get(state) is not None:
                return 0

            restored_matrix = self.restore_matrix_from_hashed_state(state)

            # Check if current state has been played before by rotating it
            for i in range(1,4):
                rotated_state = ','.join(str(int(num)) for row in np.rot90(restored_matrix,i) for num in row)
                if board_states.get(rotated_state) is not None:
                    return i
                
        # If no state has ever been played return 0 
        return 0
        

    def calculate_state_values(self, player_name):
        """
        Updates the state-value estimates for a reinforcement learning agent.

        This method updates the value of each game state encountered during the match 
        using the temporal-difference learning method. It applies a reward at the final state based on 
        the game outcome and then propagates the value updates backward through the 
        game history.

        Args:
            player_name (str): The identifier for the learning agent (e.g., "learning_agent1", "learning_agent2").
        """

        is_player1 = "1" in player_name
        board_states = self.board_states1 if is_player1 else self.board_states2
        draw_value = DRAW_VALUE_FIRST_PLAYER if is_player1 else DRAW_VALUE_SECOND_PLAYER
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
                board_states[state]['value'] += LEARNING_RATE * (board_states[old_state]['value'] - board_states[state]['value'])
                board_states[state]['next_states'].add(old_state)
                        
            old_state = state
            

    def learning_agent_move(self, player_name, exploration_rate, seed):
        """
        Executes a move for a reinforcement learning agent.

        The agent selects its move based on an exploration-exploitation trade-off:
        - With probability `exploration_rate`, the agent makes a random move.
        - Otherwise, it selects the best-known move using a greedy approach.
        
        If no valid learned move is available, the agent falls back to a random move.

        Args:
            player_name (str): The identifier for the learning agent (e.g., "learning_agent1", "learning_agent2").
            exploration_rate (float): The probability of making a random move instead of the best-known move.
            seed (int): A seed value for reproducibility in random move selection.

        Returns:
            int: Returns 0 if the agent had to resort to a random move.
        """


        np.random.seed(seed)
        random_uniform = np.random.uniform(low=0, high=1)
        if random_uniform <= exploration_rate:
            # random move
            if "1" in player_name:
                print("Entering random uniform, now doing random move")
                self.random_move(-1, seed)
            else:
                self.random_move(1, seed)
            print("RANDOM UNIFORM")

            new_state = np.zeros((3,3))

        else:
            if "1" in player_name: 
                new_state = self.choose_greedy_move(self.board_states1, -1, seed)
            else:
                new_state = self.choose_greedy_move(self.board_states2, 1, seed)

        
        # Catch case of all zero matrix meaning a random move had to be drawn
        if np.all(new_state == 0):
            return 0
        
        if "1" in player_name:
            self.placement(-1, new_state)
        else: 
            self.placement(1, new_state)
        

    def play(self,seed, player1, player2, exploration_rate):
        """
        Simulates a game of Tic-Tac-Toe between two players.

        This method alternates turns between player1 and player2 for up to 10 moves, checking 
        for a game-ending condition after each move. It supports three types of players: 
        - A random agent that makes random moves.
        - A user who provides manual input.
        - A reinforcement learning agent that makes strategic moves based on learned policies.

        If a learning agent is involved, it updates its state values at the end of the game.

        Args:
            seed (int): A seed value used for random move generation to ensure reproducibility.
            player1 (str): Specifies the type of the first player ("random_agent1", "user", "learning_agent1").
            player2 (str): Specifies the type of the second player ("random_agent2", "user", "learning_agent2").
            exploration_rate (float): The exploration rate used by learning agents to balance exploration and exploitation.
        """

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
            print("LEARNING AGENT 1 calculating state values")
            self.calculate_state_values(player1)
        if "learning" in player2:
            print("LEARNING AGENT 2 calculating state values")
            self.calculate_state_values(player2)
 
        # Reset all game instances
        self.winners_list.append(self.winner) 
        self.EndFlag = False
        self.board = np.zeros((3,3))
        self.history = list()



if __name__ == "__main__":
    agent = TicTacToeAgent()

    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    random_seed_list=random.sample(range(1,1000000), NUMBER_OF_ITERATIONS)

    for seed in random_seed_list:      
        agent.play(seed, "learning_agent1", "learning_agent2", EXPLORATION_RATE)

    with open('output_board_states2.json', 'w') as f:
        f.write(str(agent.board_states2))

    with open('output_board_states1.json', 'w') as f:
        f.write(str(agent.board_states1))
    

    with open('winner.txt', 'w') as f:
        f.write(str(agent.winners_list))


    agent.play(51, "learning_agent1", "user", 0)
    # agent.play(51, "user", "learning_agent2", 0)

