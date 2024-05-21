import os
os.environ["WEBOTS_HOME"] = "/usr/local/webots"
import numpy as np
import tensorflow as tf
from nerabotaet.state import State
from nerabotaet.robot_controller import RobotController
from ddpg import DDPGAgent
import matplotlib.pyplot as plt
import psutil


def check_memory_usage():
    # Получить процент использования оперативной памяти
    memory_percent = psutil.virtual_memory().percent
    return memory_percent



if __name__ == "__main__":
    model = DDPGAgent()
    state = State()
    controller = RobotController()
    

    # Параметры обучения
    num_episodes = 100000
    max_steps_per_episode = 10000
    rewards = []
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Эпизод')
    ax.set_ylabel('Награда')
    ax.set_title('График награды за эпизоды')

    for episode in range(num_episodes):
        # Сброс среды в начальное состояние
        memory_percent = check_memory_usage()
        controller.reset_environment()
        total_reward = 0

        for step in range(max_steps_per_episode):
            # Получение начального состояния
            current_state_data = controller.get_center_row()
            current_state_data = np.expand_dims(current_state_data, axis=0)  # Добавляем размер пакета

            # Получение действия от агента (применяется политика)
            action_probs = model.get_action(current_state_data, eval = True)


            # Получаем значения из тензора
            #action_probs = action_probs.numpy()
            if np.isnan(action_probs).any():
                # Обработка случая, когда вероятности содержат NaN
                # Например, можно заменить NaN на 0 или удалить соответствующие действия из рассмотрения
                action_probs = np.nan_to_num(action_probs, nan=0)


            # Нормализация вероятностей
            action_probs_normalized = tf.nn.softmax(action_probs).numpy()


            # Выбор действия с учетом вероятностей
            action_index = np.random.choice(len(action_probs_normalized[0]), p=action_probs_normalized[0])


            # Применение выбранного действия к среде
            if action_index == 0:
                controller.move_forward()
                reward = 0
            elif action_index == 1:
                controller.turn_left()
                reward = -1
            else:
                controller.turn_right()
                reward = -1

            # Получение следующего состояния и награды от среды
            next_state_data = controller.get_center_row()
            next_state_data = np.expand_dims(next_state_data, axis=0)

            reward += state.get_reward(current_state_data[0])

            if state.get_state(current_state_data[0], words=True) == 'crash':
                controller.reset_environment()  # Сброс среды в начальное состояние
                break

            # Обновление нейронной сети
            model.train_step(current_state_data, action_probs_normalized, reward, next_state_data, False)

            # Переход к следующему состоянию
            current_state_data = next_state_data

            # Проверка на завершение эпизода
            if step == max_steps_per_episode - 1:
                break

            total_reward += reward
        rewards.append(total_reward)
        ax.plot(rewards, 'b-')  # Обновление графика с новыми данными
        fig.canvas.draw()
        fig.canvas.flush_events()
        print(f"Эпизод {episode + 1}/{num_episodes}, Награда: {total_reward}","Memory Percent:", memory_percent)
        plt.ioff()
        plt.show(block=False)