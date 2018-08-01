import gym
import colorama as cr
import msvcrt as vc

cr.init(autoreset=True)

environment = gym.make('FrozenLake-v0')
environment.render()
environment.reset()
\
while True:
    keyIn = vc.getch().decode('utf-8').lower()
    if keyIn in ['w', 'a', 's', 'd']:
        action = 3          # up
        if keyIn == 's':
            action = 1      # down
        elif keyIn == 'a':
            action = 0      # left
        elif keyIn == 'd':
            action = 2      # right

        state, reward, isDone, information = environment.step(action)
        environment.render()
        print("state : {} \t action : {} \t reward : {} \t info : {}".format(
            state, action, reward, information))
        print() # new line
        if isDone:
            print("DONE")
            break