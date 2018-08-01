import gym
import colorama as cr
import msvcrt as vc

cr.init(autoreset=True)
gym.envs.registration.register(id='FrozenLake-v3',
entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'map_name':'4x4', 'is_slippery':False})

environment = gym.make('FrozenLake-v3')
environment.render()

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