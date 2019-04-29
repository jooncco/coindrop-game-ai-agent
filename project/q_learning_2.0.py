"""
The original code is from https://github.com/dennybritz/reinforcement-learning/tree/master/TD
"""

import sys
import numpy as np
import itertools
import pickle
from collections import defaultdict
from game import Game

# states
STAY = 0
LEFT = 1
RIGHT = 2
TIME_STAY = 3
TIME_LEFT = 4
TIME_RIGHT = 5
# bascket coverage lookup table
table_coverage = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 11]]
# time item
time_flag = 0
time_x = -1
time_distance = 0
# money
coin_distance = 0
# happiness
happy = 0

### user designed utils
def distance(basket_x, x_pos):
    dis = 0
    for i in table_coverage[basket_x]:
        dis += pow((x_pos-i), 2)
    return dis

def coin_dis(game_info):
    basket_x, item_info = game_info
    dis = 0

    for item in item_info:
        if eatable(basket_x, item) and item[0] == 1:
            dis += item[2]*distance(basket_x, item[1])
    return dis

def eatable(basket_x, item):
    if item[2] <= 1:
        return 0
    if item[1] in table_coverage[basket_x]:
        return 1
    else: #outside basket
        if abs(table_coverage[basket_x][0]-item[1]) <= 9-item[2]:
            return 1
        else:
            return 0

def happiness(game_info):
    return 50 if time_x in table_coverage[game_info[0]] else 0

###

## In our case, we have 3 actions (stay, go-left, go-right)
def get_action_num():
    return 3

## this function return policy function to choose the action based on Q value.
def make_policy(Q, epsilon, nA):
    """
    This is the epsilon-greedy policy, which select random actions for some chance (epsilon).
    (Check dennybritz's repository for detail)

    You may change the policy function for the given task.
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

## this function return state from given game information.
def get_state(counter, score, game_info):
    coins = [0, 0, 0]
    global time_flag, time_x

    # count eatable items
    for item in game_info[1]:
        if eatable(game_info[0], item):
            # time
            if item[0] == 2:
                time_flag = 1
                time_x = item[1]
            # coin
            else:
                if item[1] in table_coverage[game_info[0]]:
                    coins[STAY] += 1
                elif item[1] < table_coverage[game_info[0]][0]:
                    coins[LEFT] += 1
                else:
                    coins[RIGHT] += 1
    # return
    if time_flag:
        if time_x in table_coverage[game_info[0]]:
            return TIME_STAY
        elif time_x < table_coverage[game_info[0]][0]:
            return TIME_LEFT
        else:
            return TIME_RIGHT
    else:
        return np.argmax(coins)

## this function return reward from given previous and current score and counter.
def get_reward(prev_score, current_score, prev_counter, current_counter, game_info):
    global time_flag, time_x, time_distance, coin_distance

    if current_counter > prev_counter:
        time_flag = 0
        time_x = -1
        return 100*(current_counter - prev_counter)
    if time_flag:     # time item exists
        prev_time_dis = time_distance
        time_distance = distance(game_info[0], time_x)
        return (prev_time_dis - time_distance) + happiness(game_info)
    else:               # coins only
        prev_coin_dis = coin_distance
        coin_distance = coin_dis(game_info)
        return (current_score - prev_score) + 2*(prev_coin_dis-coin_distance)

def save_q(Q, num_episode, params, filename="model_q.pkl"):
    data = {"num_episode": num_episode, "params": params, "q_table": dict(Q)}
    with open(filename, "wb") as w:
        w.write(pickle.dumps(data))

def load_q(filename="model_q.pkl"):
    with open(filename, "rb") as f:
        data = pickle.loads(f.read())
        return defaultdict(lambda: np.zeros(3), data["q_table"]), data["num_episode"], data["params"]


def q_learning(game, num_episodes, params):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy.
    You can edit those parameters, please speficy your changes in the report.

    Args:
        game: Coin drop game environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        Q: the optimal action-value function, a dictionary mapping state -> action values.
    """

    epsilon, alpha, discount_factor = params

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(get_action_num()))

    # The policy we're following
    policy = make_policy(Q, epsilon, get_action_num())

    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        _, counter, score, game_info = game.reset()
        state = get_state(counter, score, game_info)
        action = 0

        # One step in the environment
        for t in itertools.count():
            # Take a step
            action_probs = policy(get_state(counter, score, game_info))
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            done, next_counter, next_score, game_info = game.step(action)

            next_state = get_state(counter, score, game_info)
            reward = get_reward(score, next_score, counter, next_counter, game_info)

            counter = next_counter
            score = next_score

            """
            this code performs TD Update. (Update Q value)
            You may change this part for the given task.
            """
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("Episode {}/{} (Score: {})\n".format(i_episode + 1, num_episodes, score), end="")
            sys.stdout.flush()

    return Q

def train(num_episodes, params):
    g = Game(False)
    Q = q_learning(g, num_episodes, params)
    return Q


## This function will be called in the game.py
def get_action(Q, counter, score, game_info, params):
    epsilon = params[0]
    policy = make_policy(Q, epsilon, get_action_num())
    action_probs = policy(get_state(counter, score, game_info))
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    return action

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_episode", help="# of the episode (size of training data)",
                    type=int, required=True)
    parser.add_argument("-e", "--epsilon", help="the probability of random movement, 0~1",
                    type=float, default=0.1)
    parser.add_argument("-lr", "--learning_rate", help="learning rate of training",
                    type=float, default=0.1)

    args = parser.parse_args()

    if args.num_episode is None:
        parser.print_help()
        exit(1)

    # you can pass your parameter as list or dictionary.
    # fix corresponding parts if you want to change the parameters

    num_episodes = args.num_episode
    epsilon = args.epsilon
    learning_rate = args.learning_rate

    discount_factor = 0.5
    Q = train(num_episodes, [epsilon, learning_rate, discount_factor])
    save_q(Q, num_episodes, [epsilon, learning_rate, discount_factor])

    #Q, n, params = load_q()

if __name__ == "__main__":
    main()