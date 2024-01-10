# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy as np
import time


def main():
    # Environment
    env = gym.make("Taxi-v3", render_mode='ansi')
    env.reset()

    # Training parameters for Q learning
    alpha = 0.9 # Learning rate
    gamma = 0.9 # Future reward discount factor
    num_of_episodes = 1000
    num_of_steps = 500 # per each episode
    epsilon = 0.1 # epsilon
    decay_rate= 0.005 # decay rate for epsilon

    # Q tables for rewards
    #Q_reward = -100000*np.ones((500,6)) # All same
    #Q_reward = -100000*np.random.random((500, 6)) # Random
    Q_reward = np.zeros((500,6)) # All zeros


    # First lets look at manual solution that chooses action randomly until the passenger is dropped off at the end
    # This part is only to see how the taxi environment works
    '''''
    actions = [0, 1, 2, 3, 4, 5]
    tot_reward = 0
    num_of_actions = 0
    done = False
    env.reset()
    while not done:
        # Actions: 0 south, 1 north, 2 east, 3 west, 4 pickup, 5 dropoff
        action = random.choice(actions)
        state, reward, done, truncated, info = env.step(action)
        print(env.render())
        tot_reward += reward
        num_of_actions += 1
        if done: 
            print("Total reward", tot_reward)
            print("Total number of actions", num_of_actions)
            print()
            break
    '''''
    # The total reward is (understandably) not very good this way


    # Now lets use Q-learning for training
    for episode in range(num_of_episodes):
        state = env.reset()[0]
        done = False

        for step in range(num_of_steps):
            # Explore
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            # Exploit 
            else:
                action = np.argmax(Q_reward[state, :])

            # Take action
            newstate, reward, done, truncated, info = env.step(action)

            # Q-learning update rule
            next_max = np.max(Q_reward[newstate,:]) # max Q(st+1,a)
            new_value = (1-alpha)*Q_reward[state, action] + alpha*(reward + gamma*next_max)
            Q_reward[state, action] = new_value

            # Update state to new state
            state = newstate

            if done:
                break
            
        # Lets decrease epsilon after every episode, so there is more exploring in the beginning 
        # and more exploting in the end
        epsilon = np.exp(-decay_rate*episode)
  
    # Testing 
    n_test = 10 # Testing in run 10 times
    state = env.reset()[0]
    tot_reward = 0
    tot_actions = 0
    test_tot_rewards = 0
    test_tot_actions = 0

    for test in range(n_test):
        state = env.reset()[0]
        done = False
        tot_reward = 0
        tot_actions = 0
        for t in range(50):
            action = np.argmax(Q_reward[state,:])
            state, reward, done, truncated, info = env.step(action)
            tot_reward += reward
            tot_actions += 1
            print(env.render())
            time.sleep(0.5)
            if done:
                break

        test_tot_rewards += tot_reward
        test_tot_actions += tot_actions

    # Calculate and print averages for total reward and number of actions
    print("Average total reward is", test_tot_rewards/n_test)
    print("Average number of actions is", test_tot_actions/n_test)

main()
