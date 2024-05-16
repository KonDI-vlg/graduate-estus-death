import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from networks import Agent
from robot_controller import RobotController
import time
import matplotlib.pyplot as plt

def plot_scores(episodes, scores, avg_scores):
    plt.figure(1)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(episodes, scores, label='Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Score over Episodes')
    plt.subplot(2, 1, 2)
    plt.plot(episodes, avg_scores, label='Avg Score', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Avg Score (Last 100)')
    plt.title('Average Score over Episodes')
    plt.tight_layout()
    plt.pause(0.00001)

if __name__ == '__main__':
    SIMULATION_STEP_DELAY = 500
    ROBOT_BASE_SPEED = 3

    N_EPISODES = 10000
    MAX_STEPS = 600
    LOAD_CHECKPOINT = False

    ACTOR_DIMS = [32,32]
    CRITIC_DIMS = [32,32]
    EPSILON = 0.5
    L2_FACTOR = 1e-2
    LEARNING_RATE = 1e-4
    N_ACTIONS = 3


    plt.ion()
    env = RobotController(SIMULATION_STEP_DELAY, ROBOT_BASE_SPEED)
    agent = Agent(actor_dims=ACTOR_DIMS, critic_dims=CRITIC_DIMS,
                  lr=LEARNING_RATE, n_actions=N_ACTIONS, epsilon=EPSILON, l2_factor=L2_FACTOR)

    best_score = -np.inf
    score_history = []
    #agent.load_models()

    if LOAD_CHECKPOINT:
        agent.load_models()

    start = time.time()
    for i in range(N_EPISODES):
        observation = env.reset()
        done = False
        steps = 0
        score = 0
        agent.epsilon = EPSILON * 0.995**i
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done = env.step(action)
            score += reward
            if not LOAD_CHECKPOINT:
                agent.train_step(observation, reward, next_observation, done)
            observation = next_observation
            steps += 1
            if steps >= MAX_STEPS:
                done = True
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not LOAD_CHECKPOINT:
                agent.save_models()

        if i % 100 == 0:
            n = round(time.time() - start, 0)
            print(f"Elapsed time: {int(n // 3600)} h {int(n % 3600 // 60)} m {int(n % 60)} s")

        print(f"Ep:{i + 1}/{N_EPISODES}, Score: {round(score, 2)}, Avg score: {round(avg_score, 2)}")

        # Тут где-то можешь сделать свой график

        episodes = range(1, len(score_history) + 1)
        avg_scores = [np.mean(score_history[max(0, j-99):j+1]) for j in range(len(score_history))]
        plot_scores(episodes, score_history, avg_scores)
plt.ioff()
plt.show()
