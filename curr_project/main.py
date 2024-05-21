import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from networks import Agent
from robot_controller import RobotController
#from history.params1 import *
import time
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def plot_scores(episodes, scores, avg_scores, save_path):
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
    plt.pause(0.001)
    if episodes[-1] % 100 == 0:
        print("...picture saved...")
        plt.savefig(os.path.join(save_path, 'history.png'))

if __name__ == '__main__':
    SIMULATION_STEP_DELAY = 300
    ROBOT_BASE_SPEED = 3
    SCREEN_DIVIDER = 3

    N_EPISODES = 40000
    MAX_STEPS = 600
    LOAD_CHECKPOINT = False

    AGENT_PARAMS = {
        # YOU CAN CHANGE
        "ACTOR_DIMS": [64,64],
        "CRITIC_DIMS": [64,64],
        "BATCH_SIZE": 128,
        "BUFFER_MAX_SIZE": 10000,
        "GAMMA": 0.95,
        "EPSILON": 0.7,
        "MIN_EPSILON": 0.05,
        "TAU": 0.01,
        "LEARNING_RATE": 1e-5,
        "L2_FACTOR": 1e-2,
        # DON'T CHANGE
        "N_ACTIONS": 3,
        "STATE_SHAPE": (640 // SCREEN_DIVIDER + 1 if 640 % SCREEN_DIVIDER != 0 else 640 // SCREEN_DIVIDER,),
    }
    


    plt.ion()
    env = RobotController(SIMULATION_STEP_DELAY, ROBOT_BASE_SPEED, SCREEN_DIVIDER)
    agent = Agent(AGENT_PARAMS)

    best_score = -np.inf
    score_history = []
    avg_scores = []
    actor_loss_history = []
    critic_loss_history = []

    agent.load_models()

    if LOAD_CHECKPOINT:
        agent.load_models()

    start = time.time()
    for i in range(N_EPISODES):
        observation = env.reset()
        done = False
        steps = 0
        score = 0
        if not LOAD_CHECKPOINT:
            agent.epsilon = max(AGENT_PARAMS['EPSILON'] * 0.9995**i, AGENT_PARAMS['MIN_EPSILON'])
        while not done:
            action = agent.choose_action(observation)

            next_observation, reward, done = env.step(np.argmax(action))
            steps += 1
            score += reward

            if steps >= MAX_STEPS:
                done = True
            
            if not LOAD_CHECKPOINT:
                agent.add_experience(observation,action,reward,next_observation,done)
                if agent.memory.buffer_cnt > agent.batch_size:
                    state, action, reward, next_state, done_ = agent.memory.sample_batch(AGENT_PARAMS['BATCH_SIZE'])
                    agent.train_step(state, action, reward, next_state, done_)
                

            observation = next_observation

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_scores.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score
            if not LOAD_CHECKPOINT:
                agent.save_models()

        if (i+1) % 5 == 0:
            plot_scores(range(1,i+2), score_history, avg_scores, agent.chkpt_dir)

        if i % 100 == 0:
            n = round(time.time() - start, 0)
            print(f"Elapsed time: {int(n // 3600)} h {int(n % 3600 // 60)} m {int(n % 60)} s")

        print(f"Ep:{i + 1}/{N_EPISODES}, Sc: {round(score, 2)}, Avg: {round(avg_score, 2)}")
    plt.ioff()
    plt.show()

