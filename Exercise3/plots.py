import ast
import  matplotlib.pyplot  as plt
import numpy as np

with open("winner.txt", "r") as f:
    data =  ast.literal_eval(f.read())

num_games= np.arange(0, len(data))
y_player0=[]
win0_count=0
y_player1=[]
win1_count=0
y_draw=[]
draws_count=0

for winner in data:
    if winner==1:
        win0_count+=1
    elif winner==0:
        win1_count+=1
    else:
        draws_count+=1
    y_player0.append(win0_count)
    y_player1.append(win1_count)
    y_draw.append(draws_count)

print("Percentage of win player0 ", data.count(1)/len(data)) 
print("Percentage of win player1 ", data.count(0)/len(data)) 
print("Percentage of draw ", data.count(0.2)/len(data)) 

plt.figure(figsize = (10, 6))
plt.plot(num_games,y_player0, color = 'b', linewidth = 2, label = 'player0')
plt.plot(num_games,y_player1, color = 'r', linewidth = 2, label = 'player1')
plt.plot(num_games,y_draw,  color = 'g', linewidth = 2, label = 'draw')
plt.ylabel('Number of Games')
plt.xlabel('Number of Wins')
plt.title('Learning Agent player0 VS Random player1')
plt.legend()
plt.savefig('foo.png')
plt.show()

