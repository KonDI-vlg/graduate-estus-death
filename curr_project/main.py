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

def plot_scores(episodes, scores, avg_scores, actor_losses, critic_losses, save_path):
    plt.figure(1)
    plt.clf()

    # Подграфик для очков
    plt.subplot(4, 1, 1)
    plt.plot(episodes, scores, label='Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()

    # Подграфик для средних очков
    plt.subplot(4, 1, 2)
    plt.plot(episodes, avg_scores, label='Avg Score', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Avg Score (Last 100)')
    plt.legend()

    # Новый график для потерь актера и критика на одной плоскости
    plt.subplot(4, 1, 3)
    plt.plot(episodes, actor_losses, label='Actor Loss', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(episodes, critic_losses, label='Critic Loss', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.pause(0.001)

    if episodes[-1] % 100 == 0:
        print("...picture saved...")
        plt.savefig(os.path.join(save_path, 'history.png'))

if __name__ == '__main__':
    SIMULATION_STEP_DELAY = 300
    ROBOT_BASE_SPEED = 3
    SCREEN_DIVIDER = 3

    N_EPISODES = 2000
    MAX_STEPS = 1000
    LOAD_CHECKPOINT = False

    AGENT_PARAMS = {
        # YOU CAN CHANGE
        "ACTOR_DIMS": [128,128],
        "CRITIC_DIMS": [128,128],
        "BATCH_SIZE": 128,
        "GAMMA": 0.95,
        "EPSILON": 0,
        "MIN_EPSILON": 0,
        "TAU": 0.01,
        "LEARNING_RATE": 1e-3,
        "L2_FACTOR": 1e-3,
        "DROPOUT": 0.2,
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
    CL, AL = 0, 0

    #agent.load_models()

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
                    state, action, reward, next_state, done_ = agent.memory.sample_batch(agent.batch_size)
                    CL, AL = agent.train_step(state, action, reward, next_state, done_)
                    CL = CL.numpy()
                    AL = AL.numpy()
                

            observation = next_observation

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_scores.append(avg_score)
        actor_loss_history.append(AL)
        critic_loss_history.append(AL)

        if avg_score > best_score:
            best_score = avg_score
            if not LOAD_CHECKPOINT:
                agent.save_models()

        if (i+1) % 300 == 0:
            plot_scores(range(1,i+2), score_history, avg_scores, actor_loss_history, critic_loss_history, agent.chkpt_dir)

        if i % 100 == 0:
            n = round(time.time() - start, 0)
            print(f"Elapsed time: {int(n // 3600)} h {int(n % 3600 // 60)} m {int(n % 60)} s")

        print(f"Ep:{i + 1}/{N_EPISODES}") # , Sc: {round(score, 2)}, Avg: {round(avg_score, 2)} AL: {round(AL.numpy(),2)} CL: {round(CL.numpy(),2)}
    plt.ioff()
    plt.show()

