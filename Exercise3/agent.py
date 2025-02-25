import numpy as np
import random 
from game import TicTacToe


class Agent():
    def __init__(self):
        # self.current_value
        # self.future_state
        # self.board_states={str:{'value':int, 'next_states':[]}}
        self.board_states={}
        self.alpha=0.5

    def choose_move():
        pass


    def calculate_state_values(self):
        game=TicTacToe()
        history, winner= game.play()

        for index, state in enumerate(reversed(history)):
            print("State", state)
            if index==0:
                self.board_states[state]={'value': winner,'next_states':[]}
                print("debug", self.board_states)
                old_state=state
                continue

            if self.board_states.get(state) is None:
                self.board_states[state] = {'value': 0,'next_states':[]}
            print("Value beginning", self.board_states[state]['value'])

            self.board_states[state]['value'] = (self.board_states[state]['value'] + self.alpha * (self.board_states[old_state]['value'] - self.board_states[state]['value']))
            self.board_states[state]['next_states'].append(old_state)

            print("Value afterwards", self.board_states[state]['value'])
            old_state=state

        
        
        

    def play():
        pass

if __name__ == "__main__":
    agent = Agent()
    for i in range(100):
        agent.calculate_state_values()

    print("Board states",agent.board_states)

    for index, state in enumerate(agent.board_states):

        if len(agent.board_states[state]['next_states']) >= 4:
            print("Multiple state", state, agent.board_states[state])
