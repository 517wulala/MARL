# %%
import gym
from stable_baselines3 import DQN


def main():
    env = gym.make('MountainCar-v0')  # 创建环境
    model = DQN("MlpPolicy", env, learning_rate=0.005, gamma=0.99, verbose=1)  # 修改学习率和折扣因子
    model.learn(total_timesteps=200000)  # 训练模型
    model.save("dqn_mountaincar_tuned")  # 保存模型
    # # 加载模型
    # model = DQN.load("dqn_mountaincar_tuned", env=env)
    test_model(model)  # 测试模型


def test_model(model):
    env = gym.make('MountainCar-v0', render_mode='human')  # 可视化只能在初始化时指定
    obs, _ = env.reset()
    done1, done2 = False, False
    total_reward = 0

    while not done1 or done2:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done1, done2, info = env.step(action)
        total_reward += reward

    print(f'Total Reward: {total_reward}')
    env.close()



main()