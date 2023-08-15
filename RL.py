import torch
# 用于构建神经网络的各种工具和类
import torch.nn as nn
import numpy as np
# 用于执行神经网络中的各种操作，如激活函数、池化、归一化等
import torch.nn.functional as F
import matplotlib.pyplot as plt
 
# 深度网络，全连接层
class Net(nn.Module):
    # 输入状态和动作
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        # 创建一个线性层，n_states行10列
        self.fc1 = nn.Linear(n_states, 10)
        # 创建一个线性层，10行n_actions列
        self.fc2 = nn.Linear(10, n_actions)
        # 随机初始化生成权重
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
 
    # 前向传播
    def forward(self, x):
        # 这里以一个动作为作为观测值进行输入(输入张量)
        # 线性变化后输出给10个神经元，格式：(x,x,x,x,x,x,x,x,x,x,x)
        x = self.fc1(x)
        # 激活函数，将负值设置为零，保持正值不变
        x = F.relu(x)
        # 经过10个神经元运算过后的数据，线性变化后把每个动作的价值作为输出。
        out = self.fc2(x)
        return out
 
# 定义DQN网络class
class DQN:
    #   n_states 状态空间个数；n_actions 动作空间大小
    def __init__(self, n_states, n_actions):
        print("<DQN init> n_states=", n_states, "n_actions=", n_actions)
        # 建立一个评估网络（即eval表示原来的网络） 和 Q现实网络 （即target表示用来计算Q值的网络）
        # DQN有两个net:target net和eval net,具有选动作、存储经验、学习三个基本功能
        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions)
        # 损失均方误差损失函数
        self.loss = nn.MSELoss()
        # 优化器，用于优化评估神经网络更新模型参数（仅优化eval），使损失函数尽量减小
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.01)
        self.n_actions = n_actions  #   状态空间个数
        self.n_states = n_states    #   动作空间大小
 
        # 使用变量
        # 用来记录学习到第几步了
        self.learn_step_counter = 0  # target网络学习计数
        # 用来记录当前指到数据库的第几个数据了
        self.memory_counter = 0  # 记忆计数
        # 创建一个2000行6列的矩阵，即表示可存储2000行经验，每一行6个特征值
        # 2*2表示当前状态state(x,y)和下一个状态next_state(x,y) + 1表示选择一个动作 + 1表示一个奖励值
        self.memory = np.zeros((2000, 2 * 2 + 1 + 1))
        self.cost = []  # 记录损失值
        self.steps_of_each_episode = []  # 记录每轮走的步数
 
    # 进行选择动作
    # x是state= [-0.5 -0.5]
    def choose_action(self, x, epsilon):
        # 扩展一行,因为网络是多维矩阵,输入是至少两维
        # torch.FloatTensor(x)先将x转化为浮点数张量
        # torch.unsqueeze(input, dim)再将一维的张量转化为二维的,dim=0时数据为行方向扩，dim=1时为列方向扩
        # 例如 [1.0, 2.0, 3.0] -> [[1.0, 2.0, 3.0]]
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # 在大部分情况，我们选择 去max-value
        if np.random.uniform() < epsilon:   # greedy # 随机结果是否大于EPSILON（0.9）
            # 获取动作对应的价值
            action_value = self.eval_net.forward(x)
            #   torch.max() 返回输入张量所有元素的最大值，torch.max(input, dim)，dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
            #   torch.max(a, 1)[1] 代表a中每行最大值的索引
            # .data.numpy()[0]将Variable转换成tensor
            # 哪个神经元值最大，则代表下一个动作
            action = torch.max(action_value, 1)[1].data.numpy()[0]
        # 在少部分情况，我们选择 随机选择 （变异）
        else:
            #   random.randint(参数1，参数2)函数用于生成参数1和参数2之间的任意整数，参数1 <= n < 参数2
            action = np.random.randint(0, self.n_actions)
        return action


    # 存储经验
    # 存储【本次状态，执行的动作，获得的奖励分，完成动作后产生的下一个状态】
    def store_transition(self, state, action, reward, next_state):
        # 把所有的记忆捆在一起，以 np 类型
        # 把 三个矩阵 s ,[a,r] ,s_  平铺在一行 [a,r]是因为 他们都是 int 没有 [] 就无法平铺 ，并不代表把他们捆在一起了
        #  np.hstack()是把矩阵按水平方向堆叠数组构成一个新的数组
        transition = np.hstack((state, [action, reward], next_state))
        # index 是 这一次录入的数据在 MEMORY_CAPACITY 的哪一个位置
        # 如果记忆超过上线，我们重新索引。即覆盖老的记忆。
        index = self.memory_counter % 2000
        self.memory[index, :] = transition  # 将transition添加为memory的一行
        self.memory_counter += 1


    # 从存储学习数据
    # target_net是达到次数后更新， eval_net是每次learn就进行更新
    def learn(self):
        # 更新 target_net
        if self.learn_step_counter % 100 == 0:
            # 将评估网络的参数状态复制到目标网络中
            # 即将target_net网络变成eval_net网络，实现模型参数的软更新
            self.target_net.load_state_dict((self.eval_net.state_dict()))
        self.learn_step_counter += 1
 
        # eval net是 每次learn 就进行更新
        # 从[0,200)中随机抽取16个数据并组成一维数组，该数组表示记忆索引值
        sample_index = np.random.choice(200, 16)
        # 表示从 self.memory 中选择索引为 sample_index 的行，: 表示选取所有列
        # 即得到16组随机选取的训练样本。
        memory = self.memory[sample_index, :]
        # 从记忆当中获取[0,2)列，即第零列和第一列，表示状态特征
        state = torch.FloatTensor(memory[:, :2])
        # 从记忆中获取[2,3)列，即第二列，表示动作特征
        action = torch.LongTensor(memory[:, 2:3])
        # 从记忆中获取[3,4)列，即第三列，表示奖励特征
        reward = torch.LongTensor(memory[:, 3:4])
        # 从记忆中获取[4,5)列，即第四列和第五列，表示下一个状态特征
        next_state = torch.FloatTensor(memory[:, 4:6])

        # 获得当前状态的动作对应的预测Q值
        # self.eval_net(state)表示输入当前state，通过forward()函数输出状态对应的Q值估计
        # .gather(1, action)表示从上述Q值估计的集合中，第一个维度上获取action对应的的Q值
        # 将Q值赋值给q_eval，表示所采取动作的预测value
        q_eval = self.eval_net(state).gather(1, action)

        # 获得下一步状态的Q值
        # 把target网络中下一步的状态对应的价值赋值给q_next；此处有时会反向传播更新target，但此处不需更新，故加.detach()
        q_next = self.target_net(next_state).detach()

        # 计算对于的最大价值
        # q_target 实际价值的计算  ==  当前价值 + GAMMA（未来价值递减参数） * 未来的价值
        # max函数返回索引的最大值
        # unsqueeze(1)将上述计算出来的最大 Q 值的张量在第 1 个维度上扩展一个维度，变为一个列向量。
        q_target = reward + 0.9 * q_next.max(1)[0].unsqueeze(1)

        # 通过预测值与真实值计算损失 q_eval预测值， q_target真实值
        loss = self.loss(q_eval, q_target)
        self.cost.append(loss.detach().numpy())
        # 根据误差，去优化我们eval网, 因为这是eval的优化器
        # 反向传递误差，进行参数更新
        self.optimizer.zero_grad()  # 梯度重置
        loss.backward()  # 反向求导
        self.optimizer.step()  # 更新模型参数

    # 绘制损失图
    def plot_cost(self):
        # np.arange(3)产生0-2数组
        plt.plot(np.arange(len(self.cost)), self.cost)
        plt.xlabel("step")
        plt.ylabel("cost")
        plt.show()

    # 绘制每轮需要走几步
    def plot_steps_of_each_episode(self):
        plt.plot(np.arange(len(self.steps_of_each_episode)), self.steps_of_each_episode)
        plt.xlabel("episode")
        plt.ylabel("done steps")
        plt.show()