'''
@Author ：YZX
@Date ：2023/8/7 15:47 
@Python-Version ：3.8
'''

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 深度网络，全连接层
class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        # n_states状态个数
        self.fc1 = nn.Linear(n_states, 10)
        # n_actions动作个数
        self.fc2 = nn.Linear(10, n_actions)
        self.fc1.weight.data.normal_(0, 0.1)  # 初始化权重，用二值分布来随机生成参数的值
        self.fc2.weight.data.normal_(0, 0.1)

    # 前向传播
    def forward(self, x):
        # 这里以一个动作为作为观测值进行输入，然后把他们输出给10个神经元
        x = self.fc1(x)
        # 激活函数
        x = F.relu(x)
        # 经过10个神经元运算过后的数据， 把每个动作的价值作为输出。
        out = self.fc2(x)
        return out


# 定义DQN 网络class
class DQN:
    #   n_states 状态空间个数；n_actions 动作空间大小
    def __init__(self, n_states, n_actions):
        print("<DQN init> n_states=", n_states, "n_actions=", n_actions)
        # 建立一个评估网络（eval） 和 Q现实网络 （target）
        # DQN有两个net:target net和eval net,具有选动作，存经历，学习三个基本功能
        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions)
        # 损失函数
        self.loss = nn.MSELoss()
        # 优化器，优化评估神经网络（仅优化eval）
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.01)
        self.n_actions = n_actions  # 状态空间个数
        self.n_states = n_states  # 动作空间大小

        # 使用变量
        # 用来记录学习到第几步了
        self.learn_step_counter = 0  # target网络学习计数
        # 用来记录当前指到数据库的第几个数据了
        self.memory_counter = 0  # 记忆计数
        # MEMORY_CAPACITY = 2000 ， 限制了数据库只能记住2000个。前面的会被后面的覆盖
        # 一次存储的数据量有多大   MEMORY_CAPACITY 确定了memory数据库有多大 ，
        # 后面的 N_STATES * 2 + 2 是因为 两个 N_STATES（在这里是4格子，因为N_STATES就为4）  + 一个 action动作（1格） + 一个 rward（奖励）
        self.memory = np.zeros((2000, 2 * 2 + 2))  # 2*2(state和next_state,每个x,y坐标确定)+2(action和reward),存储2000个记忆体
        self.cost = []  # 记录损失值
        self.steps_of_each_episode = []  # 记录每轮走的步数

    # 进行选择动作
    # x是state= [-0.5 -0.5]
    def choose_action(self, x, epsilon):
        # print("<choose_action> x=", x, "torch.FloatTensor(x)=", torch.FloatTensor(x))
        # 获取输入
        # torch.unsqueeze(input, dim) → Tensor unsqueeze()函数起升维的作用,参数dim表示在哪个地方加一个维度，注意dim范围在:[-input.dim() - 1, input.dim() + 1]之间，比如输入input是一维，则dim=0时数据为行方向扩，dim=1时为列方向扩，再大错误。
        # torch.FloatTensor(x)= tensor([-0.5000, -0.5000])
        # 扩展一行,因为网络是多维矩阵,输入是至少两维
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # tensor([[-0.5000, -0.5000]])
        # print("<choose_action> x=", x, "epsilon=", epsilon)
        # 在大部分情况，我们选择 去max-value
        if np.random.uniform() < epsilon:  # greedy # 随机结果是否大于EPSILON（0.9）
            action_value = self.eval_net.forward(x)
            #   torch.max() 返回输入张量所有元素的最大值，torch.max(input, dim)，dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
            #   torch.max(a, 1)[1] 代表a中每行最大值的索引
            # .data.numpy()[0]将Variable转换成tensor
            # 哪个神经元值最大，则代表下一个动作
            action = torch.max(action_value, 1)[1].data.numpy()[0]
            # <choose_action> action_value= tensor([[-0.2394, -0.3109, -0.3330, -0.0376]], grad_fn=<AddmmBackward0>) torch.max(action_value, 1)= torch.return_types.max(values=tensor([-0.0376], grad_fn=<MaxBackward0>), indices=tensor([3])) torch.max(action_value, 1)[1]= tensor([3]) action= 3
            # print("<choose_action> action_value=", action_value, "torch.max(action_value, 1)=",torch.max(action_value, 1),"torch.max(action_value, 1)[1]=",torch.max(action_value, 1)[1], "action=", action)
        # 在少部分情况，我们选择 随机选择 （变异）
        else:
            #   random.randint(参数1，参数2)函数用于生成参数1和参数2之间的任意整数，参数1 <= n < 参数2
            action = np.random.randint(0, self.n_actions)
        # print("action=", action)
        return action

    # 存储数据
    # 本次状态，执行的动作，获得的奖励分， 完成动作后产生的下一个状态。
    # 存储这四个值
    def store_transition(self, state, action, reward, next_state):

        # 把所有的记忆捆在一起，以 np类型
        # 把 三个矩阵 s ,[a,r] ,s_  平铺在一行 [a,r]是因为 他们都是 int 没有 [] 就无法平铺 ，并不代表把他们捆在一起了
        #  np.vstack()是把矩阵进行列连接
        transition = np.hstack((state, [action, reward], next_state))
        # state= [0.25 0.  ] action= 3 reward= 1 next_state= [0. 0.]
        # <store_transition> transition= [0.25 0.   3.   1.   0.   0.  ]
        # print("<store_transition> transition=", transition)

        # index 是 这一次录入的数据在 MEMORY_CAPACITY 的哪一个位置
        index = self.memory_counter % 200  # 满了就覆盖旧的
        # 如果，记忆超过上线，我们重新索引。即覆盖老的记忆。
        self.memory[index, :] = transition  # 将transition添加为memory的一行
        # print("<store_transition> memory=", self.memory)
        self.memory_counter += 1

    # 从存储学习数据
    # target 是 达到次数后更新， eval net是 每次learn 就进行更新
    def learn(self):
        # print("<learn>")
        # target net 更新频率,用于预测，不会及时更新参数
        # target parameter update  是否要更新现实网络
        # target Q现实网络 要间隔多少步跟新一下。 如果learn步数 达到 TARGET_REPLACE_ITER  就进行一次更新
        if self.learn_step_counter % 100 == 0:
            # 把最新的eval 预测网络 推 给target Q现实网络
            # 也就是变成，还未变化的eval网
            self.target_net.load_state_dict((self.eval_net.state_dict()))
            # 'fc1.weight', 'fc1.bias', 'fc2.weight', ....
            # print("<learn> eval_net.state_dict()=", (self.eval_net.state_dict()))
        self.learn_step_counter += 1

        #  eval net是 每次learn 就进行更新
        #  更新逻辑就是从记忆库中随机抽取BATCH_SIZE个（32个）数据。
        # 使用记忆库中批量数据
        # sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # 从 数据库中 随机 抽取 BATCH_SIZE条数据
        sample_index = np.random.choice(200, 16)  # 200个中随机抽取16个作为batch_size
        # sample_index= [ 34  48 153  60   5 140  74  81  93  85 138  33 118  90  11 124]
        # print("<learn> sample_index=", sample_index)
        memory = self.memory[sample_index, :]  # 抽取BATCH_SIZE个（16个）个记忆单元， 把这BATCH_SIZE个（16个）数据打包.
        # memory= [[-0.5  -0.25  1.    0.   -0.5   0.  ]
        #          [-0.5   0.25  1.    0.   -0.5   0.25]
        #          [ 0.    0.25  2.    0.    0.25  0.25]
        #  ...]
        # print("<learn> memory=", memory)

        state = torch.FloatTensor(memory[:, :2])  # # 32个记忆的包，包里是（当时的状态） 所有行里取0,1

        # 下面这些变量是 32个数据打包的变量
        #   state= tensor([[-0.5000, -0.2500],
        #         [-0.5000,  0.2500],
        #         [ 0.0000,  0.2500],
        #   ...]
        # print("<learn> state=", state)
        action = torch.LongTensor(memory[:, 2:3])  # # 32个记忆的包，包里是（当时做出的动作）2
        #   action= tensor([[1],
        #         [1],
        #         [2],
        #   ...]
        # print("<learn> action=", action)
        reward = torch.LongTensor(memory[:, 3:4])  # # 32个记忆的包，包里是 （当初获得的奖励）3
        next_state = torch.FloatTensor(memory[:, 4:6])  # 32个记忆的包，包里是 （执行动作后，下一个动作的状态）4,5

        # q_eval w.r.t the action in experience
        # q_eval的学习过程
        # self.eval_net(state).gather(1, action)  输入我们包（32条）中的所有状态 并得到（32条）所有状态的所有动作价值，
        # .gather(1,action) 只取这32个状态中 的 每一个状态的最大值
        # 预期价值计算 ==  随机32条数据中的最大值
        # 计算loss,
        # q_eval:所采取动作的预测value,
        # q_target:所采取动作的实际value
        # a.gather(0, b)分为3个部分，a是需要被提取元素的矩阵，0代表的是提取的维度为0，b是提取元素的索引。
        # 当前状态的预测：
        # 输入现在的状态state，通过forward()生成所有动作的价值，根据价值选取动作，把它的价值赋值给q_eval
        q_eval = self.eval_net(state).gather(1, action)  # eval_net->(64,4)->按照action索引提取出q_value

        #  state= tensor([[-0.2500, -0.2500],
        #         [-0.2500, -0.2500],
        #         [-0.5000, -0.5000],
        # ...]
        # eval_net(state)= tensor([[-0.1895, -0.2704, -0.3506, -0.3678],
        #         [-0.1895, -0.2704, -0.3506, -0.3678],
        #         [-0.2065, -0.2666, -0.3501, -0.3738],
        # ...]
        #  action= tensor([[0],
        #         [1],
        #         [0],
        # ...]
        # q_eval= tensor([[-0.1895],
        #         [-0.2704],
        #         [-0.2065],
        # ...]
        # print("<learn> eval_net(state)=", self.eval_net(state), "q_eval=", q_eval)

        # 下一步状态的预测：
        # 计算最大价值的动作：输入下一个状态 进入我们的现实网络 输出下一个动作的价值  .detach() 阻止网络反向传递，我们的target需要自己定义该如何更新，它的更新在learn那一步
        # 把target网络中下一步的状态对应的价值赋值给q_next；此处有时会反向传播更新target，但此处不需更新，故加.detach()
        q_next = self.target_net(next_state).detach()  # detach from graph, don't backpropagate
        # 计算对于的最大价值
        # q_target 实际价值的计算  ==  当前价值 + GAMMA（未来价值递减参数） * 未来的价值
        # max函数返回索引加最大值，索引是1最大值是0 torch.max->[values=[],indices=[]] max(1)[0]->values=[]
        q_target = reward + 0.9 * q_next.max(1)[0].unsqueeze(1)  # label # shape (batch, 1)
        #   q_next= tensor([[-0.1943, -0.2676, -0.3566, -0.3752],
        #                   [-0.1848, -0.2731, -0.3446, -0.3604],
        #                   [-0.2065, -0.2666, -0.3501, -0.3738],
        #                   ...]
        #   q_next.max(1)= torch.return_types.max(values=tensor([-0.1943, -0.1848, -0.2065,...]), indices=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        #   q_next.max(1)[0]= tensor([-0.1943, -0.1848, -0.2065,....])
        #   q_next.max(1)[0].unsqueeze(1)= tensor([[-0.1943], [-0.1848], [-0.2065],...])
        #   q_target= tensor([[-0.1749],
        #         [-0.1663],
        #         [-0.1859],
        #   ...]
        # print("<learn> q_target=", q_target, "q_next=", q_next, "q_next.max(1)=", q_next.max(1), "q_next.max(1)[0]=", q_next.max(1)[0], "q_next.max(1)[0].unsqueeze(1)=", q_next.max(1)[0].unsqueeze(1))
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
        #   loss = self.loss(q_eval, q_target)
        #   self.cost.append(loss)
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