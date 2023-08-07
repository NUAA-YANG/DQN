'''
@Author ：YZX
@Date ：2023/8/7 15:49 
@Python-Version ：3.8
'''

from MazeEnv import Maze
from RL import DQN
import time


# 注释参考https://www.pancake2021.work/?m=202207

def run_maze():
    print("====Game Start====")

    step = 0  # 已进行多少步
    max_episode = 500  # 总共需要进行多少轮，每轮有多步

    for episode in range(max_episode):

        # 每一次新的训练
        # 开始，会重置我们的env， 每一次训练的环境都是独立的而完全一样的，只有网络记忆是一直留存的
        state = env.reset()  # 重置智能体位置，# 获得初始化 observation 环境特征

        step_every_episode = 0  # 本轮已进行多少步
        epsilon = episode / max_episode  # parameter for epsilon greedy policy，动态变化随机值

        # 开始实验循环
        # 只有env认为 这个实验死了，才会结束循环
        while True:
            if episode < 10:
                time.sleep(0.1)
            if episode > 480:
                time.sleep(0.5)

            # 刷新环境状态  ， 使得screen 可以联系的动
            env.render()  # 显示新位置
            # 根据 输入的环境特征s  输出选择动作 a
            action = model.choose_action(state, epsilon)  # 根据状态选择行为
            # 通过当前选择的动作得到，执行这个动作后的结果也就是，下一步状态s_（也就是observation） 特征值矩阵  ，
            # 立即回报r 返回动作执行的奖励 ， r是一个float类型
            # 终止状态 done （done=True时环境结束） ， done 是 bool
            # 调试信息 info （一般没用）
            # 环境根据行为给出下一个状态，奖励，是否结束。
            next_state, reward, terminal = env.step(action)  # env.step(a) 是执行 a 动作
            # state = [-0.5 - 0.5]  action = 1  reward = 0  next_state = [-0.5 - 0.25] terminal = False
            # print("episode=", episode,"step=", step_every_episode, "state=", state, "action=", action, "reward=", reward, "next_state=", next_state, "terminal=", terminal)
            # 到这里，预测流程就结束........

            # 存储数据
            # 每完成一个动作，记忆存储数据一次
            model.store_transition(state, action, reward, next_state)  # 模型存储经历

            # 按批更新
            # 假如我们总训练2000次，
            # 在训练第i_episode（200）次后，我们数据库中累计的信息超过3000条后。
            # 这个时 dqn中的数据库中的记忆条数  大于 数据库的容量
            # 控制学习起始时间(先积累记忆再学习)和控制学习的频率(积累多少步经验学习一次)
            if step > 200 and step % 5 == 0:
                # 它就会开对去学习。
                # eval 每学一次就会更新一次  # 它的更新思路是从我历史记忆中随机抽取数据。 #学习一次，就在数据库中随机挑选BATCH_SIZE（32条） 进行打包
                # 而target不一样，它是在我们学习过程中到一定频率（TARGET_REPLACE_ITER，来决定）。它的思路是：target网会去复制eval网的参数
                model.learn()

            # env判断游戏没有结束进行while循环，下次状态变成当前状态， 开始走下一步。
            state = next_state  # 当前状态

            # 在满足 大于数据库容量的条件下，我再看env.step(a) 返回的done，env是否认为实验结束了
            # 状态是否为终止
            if terminal:
                print("episode=", episode, end=",")  # 第几轮
                print("step=", step_every_episode)  # 第几步
                model.steps_of_each_episode.append(step_every_episode)  # 记录每轮走的步数
                break

            step += 1  # 总步数+1
            step_every_episode += 1  # 当前轮的步数+1

    # 游戏环境结束
    print("====Game Over====")
    env.destroy()


if __name__ == "__main__":
    env = Maze()  # 环境
    # 实例化DQN类，也就是实例化这个强化学习网络
    model = DQN(
        n_states=env.n_states,  # 状态空间个数
        n_actions=env.n_actions  # 动作空间个数
    )  # 算法模型
    run_maze()  # 训练

    env.mainloop()  # mainloop()方法允许程序循环执行,并进入等待和处理事件
    model.plot_cost()  # 画误差曲线
    model.plot_steps_of_each_episode()  # 画每轮走的步数