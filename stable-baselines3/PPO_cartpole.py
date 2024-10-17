# %%
# 使用rl-sb3环境
import gym
from stable_baselines3 import PPO

env = gym.make('CartPole-v1')  # 创建环境
model = PPO("MlpPolicy", env, verbose=1)  # 创建模型
model.learn(total_timesteps=20000)  # 训练模型
model.save("ppo_cartpole")  # 保存模型
#test_model(model)  # 测试模型

'''
运行test_model(model)的时候会报溢出错误
def test_model(model):
    env = gym.make('CartPole-v1', render_mode='human')  # 可视化只能在初始化时指定
    obs, _ = env.reset()
    done1, done2 = False, False
    total_reward = 0

    while not done1 or done2:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done1, done2, info = env.step(action)
        total_reward += reward

    print(f'Total Reward: {total_reward}')
    env.close()
'''
# %%

def test_model(model):
    # 创建环境，指定渲染模式为'human'，以便可视化
    env = gym.make('CartPole-v1', render_mode='human')
    obs = env.reset()
    total_reward = 0
    
    # 持续交互直到环境返回done=True
    while True:
        # 预测动作，设置deterministic=True以获取确定性行为
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        # 如果环境返回done=True，则打印总奖励并重置环境
        if done:
            print(f'Total Reward: {total_reward}')
            total_reward = 0  # 重置总奖励
            obs = env.reset()  # 重置环境以开始新的episode
            break  # 如果只需要一个episode，可以用break退出循环

    # 关闭环境
    env.close()

test_model(model)
# %%
