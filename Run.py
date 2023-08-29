'''
@Author ：YZX
@Date ：2023/8/7 16:03
@Python-Version ：3.8
'''

from MazeEnv import Maze
from RL import DQN
import time

 
def run_maze():
    print("====Game Start====")
 
    step = 0    # 已进行多少步
    max_episode = 500   # 总共需要进行多少轮
 
    for episode in range(max_episode):

        # 环境和位置重置，但是memory一直保留
        state = env.reset()

        # 本轮已进行多少步
        step_every_episode = 0
        # 动态变化随机值
        epsilon = episode / max_episode
 
        # 开始实验循环
        # 只有env认为 这个实验死了，才会结束循环
        while True:
            if episode < 10:
                time.sleep(0.1)
            if episode > 480:
                time.sleep(0.2)
 
            # 刷新环境状态，显示新位置
            env.render()
            # 根据输入的环境特征 s  输出选择动作 a
            action = model.choose_action(state, epsilon)  # 根据状态选择行为
            # 环境根据行为给出下一个状态，奖励，是否结束。
            next_state, reward, terminal = env.step(action) # env.step(a) 是执行 a 动作
            # 每完成一个动作，记忆存储数据一次
            model.store_transition(state, action, reward, next_state)  # 模型存储经历

            # 按批更新
            if step > 200 and step % 5 == 0:
                model.learn()
 
            # 状态转变
            state = next_state

            # 状态是否为终止
            if terminal:
                print("episode=", episode, end=",") # 第几轮
                print("step=", step_every_episode)  # 第几步
                model.steps_of_each_episode.append(step_every_episode) # 记录每轮走的步数
                break
 
            step += 1   # 总步数+1
            step_every_episode += 1 # 当前轮的步数+1
 
    # 游戏环境结束
    print("====Game Over====")
    env.destroy()
 
 
if __name__ == "__main__":
    env = Maze()  # 环境
    # 实例化DQN类，也就是实例化这个强化学习网络
    model = DQN(n_states=env.n_states,n_actions=env.n_actions)
    run_maze()  # 训练
 
    env.mainloop()  # mainloop()方法允许程序循环执行,并进入等待和处理事件
    model.plot_cost()  # 画误差曲线
    model.plot_steps_of_each_episode()  # 画每轮走的步数