SIMULATION_STEP_DELAY = 300
ROBOT_BASE_SPEED = 3
SCREEN_DIVIDER = 3

N_EPISODES = 1000
MAX_STEPS = 1000
LOAD_CHECKPOINT = False

AGENT_PARAMS = {
    # YOU CAN CHANGE
    "ACTOR_DIMS": [64,64],
    "CRITIC_DIMS": [64,64],
    "BATCH_SIZE": 64,
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