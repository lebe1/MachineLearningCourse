import numpy as np
import random 
from game import TicTacToe


class Agent():
    def __init__(self, exploration_rate):
        # self.current_value
        # self.future_state
        # self.board_states={str:{'value':int, 'next_states':[]}}
        self.row_column_index = {1: [0,0]}
        self.board_states={}
        self.alpha=0.5
        self.exploration = exploration_rate
        #self.seed=seed

    def choose_move(self, game, current_state):
        np.random.seed(42)
        if np.random.uniform(low=0, high=1) <= self.exploration:
            # random move
            game.move(-1)

        else:
            # greedy move according to best value

            best_value = -1000000
            best_state = ""

            # make random move if the current state hasn't been encountered yet
            if self.board_states.get(current_state) is None:
                game.move(-1)
                return np.zeros((3,3))
            else:
                for next_state in self.board_states[current_state]["next_states"]:
                    if (self.board_states[next_state]["value"] > best_value):
                        best_state = next_state
                        best_value = self.board_states[next_state]["value"]

            matrix_best_state = []

            if best_state == "":
                print("ERROR", self.board_states[current_state])
                print(current_state)
            best_state = best_state.split(",")

            for idx, letter in enumerate(best_state):
                matrix_best_state.append(int(letter))

            matrix_best_state = np.array(matrix_best_state).reshape(3, 3)
            return matrix_best_state
        


    def calculate_state_values(self, game):
        #game=TicTacToe()
        #history, winner= game.play()

        for index, state in enumerate(reversed(game.history)):
            print("State", state)
            if index==0:
                self.board_states[state]={'value': game.winner,'next_states': set()}
                #print("debug", self.board_states)
                old_state=state
                continue

            if self.board_states.get(state) is None:
                self.board_states[state] = {'value': 0,'next_states': set()}
            print("Value beginning", self.board_states[state]['value'])

            self.board_states[state]['value'] = (self.board_states[state]['value'] + self.alpha * (self.board_states[old_state]['value'] - self.board_states[state]['value']))
            self.board_states[state]['next_states'].add(old_state)

            print("Value afterwards", self.board_states[state]['value'])
            old_state=state

        
        
    def play(self):
        game = TicTacToe()

        for i in range(9):
            if game.EndFlag:
                game.history.append(','.join(str(int(num)) for row in game.board for num in row))
                break
            else:   
                if i % 2 == 0:
                    print("AGENT IS MOVING")
                    current_state = ','.join(str(int(num)) for row in game.board for num in row)
                    print("CURRENT STATE: ")
                    print(current_state)
                    print("CURRENT BOARD: ")
                    print(game.board)
                    new_state = self.choose_move(game, current_state)
                    if np.all(new_state == 0):
                        continue
                    print("NEW STATE (AFTER MOVE): ")
                    print(new_state)
                    game.placement(-1, new_state)
                else:
                    print("PLAYER RANDOM IS MOVING")
                    game.move(1)

        return game


if __name__ == "__main__":
    agent = Agent(exploration_rate=0.2)

    for i in range(10):
        game = agent.play()
        agent.calculate_state_values(game)
        print(agent.board_states)


