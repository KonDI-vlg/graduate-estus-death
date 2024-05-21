import os
os.environ["WEBOTS_HOME"] = "/usr/local/webots"
class State:
    def __init__(self):
        self.state_dict = {"crash": 0, "near": 1, "far": 2}

    def get_state(self, data, words=False):
        min_value = min(data)

        if words:
            if min_value <= 0.19:
                return 'crash'
            elif 0.2 <= min_value <= 0.3:
                return 'near'
            else:
                return 'far'
        else:
            if min_value <= 0.19:
                return self.state_dict['crash']
            elif 0.2 <= min_value <= 0.3:
                return self.state_dict['near']
            else:
                return self.state_dict['far']

    def get_reward(self, data):
        min_value = min(data)

        if min_value <= 0.19:
            return -min_value * 100
        elif 0.2 <= min_value <= 0.4:
            return -(1 / min_value)
        else:
            return 1


if __name__ == "__main__":
    state = State()
    print(state.get_state([i for i in range(1, 11)], words=False))
    print(state.get_reward([i for i in range(1, 11)]))