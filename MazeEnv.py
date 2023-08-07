'''
@Author ：YZX
@Date ：2023/8/7 15:44 
@Python-Version ：3.8
'''

import tkinter as tk
import sys
import numpy as np

UNIT = 40  # pixels 像素
MAZE_H = 4  # grid height y轴格子数
MAZE_W = 4  # grid width x格子数


# 迷宫
class Maze(tk.Tk, object):
    def __init__(self):
        print("<env init>")
        super(Maze, self).__init__()

        # 动作空间(定义智能体可选的行为),action=0-3
        self.action_space = ['u', 'd', 'l', 'r']
        # 使用变量
        self.n_actions = len(self.action_space)
        # 状态空间，state=0,1
        self.n_states = 2

        # 配置信息
        self.title('maze')
        # 设置屏幕大小
        self.geometry("160x160")

        # 初始化操作
        self.__build_maze()

    # 渲染画面
    def render(self):
        # time.sleep(0.1)
        self.update()

    # 重置环境
    def reset(self):
        # 智能体回到初始位置
        # time.sleep(0.1)
        self.update()
        self.canvas.delete(self.rect)

        origin = np.array([20, 20])

        # 智能体位置，前两个左上角坐标(x0,y0)，后两个右下角坐标(x1,y1)
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # return observation 状态
        # canvas.coords(长方形/椭圆),会得到 【左极值点、上极值点、右极值点、下极值点】这四个点组成的元组，:2表示前2个
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (
                    MAZE_H * UNIT)

    # 智能体向前移动一步：返回next_state,reward,terminal
    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        next_coords = self.canvas.coords(self.rect)  # next state

        # reward function
        if next_coords == self.canvas.coords(self.oval):
            reward = 1
            print("victory")
            done = True
        elif next_coords in [self.canvas.coords(self.hell1)]:
            reward = -1
            print("defeat")
            done = True
        else:
            reward = 0
            done = False
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (MAZE_H * UNIT)
        return s_, reward, done

    def __build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        origin = np.array([20, 20])
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        # 陷阱
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        oval_center = origin + UNIT * 2
        # 出口
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')
        # 智能体
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        self.canvas.pack()
