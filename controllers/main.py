import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from networks import Agent
from robot_controller import RobotController
import time

if __name__ == '__main__':
    SIMULATION_STEP_DELAY = 1000
    N_EPISODES = 10000
    MAX_STEPS = 300
    EPSILON = 0.7
    L2_FACTOR = 1e-6
    LEARNING_RATE = 0.0025
    N_ACTIONS = 3
    LOAD_CHECKPOINT = False

    env = RobotController(SIMULATION_STEP_DELAY)
    agent = Agent(lr=LEARNING_RATE, n_actions=N_ACTIONS, epsilon=EPSILON, l2_factor=L2_FACTOR)

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
        agent.epsilon = EPSILON * 0.99**i
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done = env.step(action,steps)
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
        """
        if i % 100 == 0:
            ...

        """
