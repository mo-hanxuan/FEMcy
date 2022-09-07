# -*- coding: utf-8 -*-

# -------------------------------------------
# Module
# get the color by a specific mod
# -------------------------------------------

import warnings


def getColor(x, mod=4):
    delta = 1.e-3
    if not (0. - delta <= x <= 1. + delta):
        red, green, blue = 0.2, 0.2, 0.2
        if x > 1. + delta:
            red, green, blue = 0.5, 0.5, 0.5
            warnings.warn('colorBar x > 1.')
        elif x < 0. - delta:
            red, green, blue = 0.2, 0.2, 0.2
            warnings.warn('colorBar x < 0.')
        return red, green, blue
    def case1():
        # (1.0 red -> 0.5 green -> 0.0 blue)
        if x >= 0.5:
            red = (x - 0.5) / 0.5
            green = (1. - x) / 0.5
            blue = 0.
        else:
            red = 0.
            green = x / 0.5
            blue = (0.5 - x) / 0.5
        return red, green, blue
    def case2():
        # (1, 0, 0) -> (0.5, 1, 0.5) -> (0, 0, 1)
        #   red     ->  bright green -> blue (more smooth)
        red = x
        blue = 1. - x
        green = (1. - x) / 0.5 if x >= 0.5 else x / 0.5
        return red, green, blue
    def case3():
        # (1.0 red -> 0.5 white -> 0.0 blue)
        if x >= 0.5:
            red = 1.
            green = (1. - x) / 0.5
            blue = (1. - x) / 0.5
        else:
            red = x / 0.5
            green = x / 0.5
            blue = 1.
        return red, green, blue
    def case4():
        # rainbow colorBar, 4 intervals,
        #      (1 ~  0.75  ~ 0.5   ~ 0.25 ~ 0)
        # -> (red ~ yellow ~ green ~ cyan ~ blue)
        if x >= 0.75:
            red = 1.
            green = (1. - x) / 0.25
            blue = 0.
        elif 0.5 <= x < 0.75:
            red = (x - 0.5) / 0.25
            green = 1.
            blue = 0.
        elif 0.25 <= x < 0.5:
            red = 0.
            green = 1.
            blue = (0.5 - x) / 0.25
        else:
            red = 0.
            green = x / 0.25
            blue = 1.
        return red, green, blue
    def case5():
        # (1.0 red -> 0.0 blue)  smooth across 0 -> 1
        # (1,0,0) -> (0.5,0,0.5) -> (0,0,1)
        #   red   ->    purple   ->  blue
        red = x
        green = 0.
        blue = 1. - x
        return red, green, blue
    def case6():
        # (1.0 red -> 0.5 black -> 0.0 blue)
        if x >= 0.5:
            red = (x - 1. / 2.) / (1. / 2.)
            green = 0.
            blue = 0.
        else:
            red = 0.
            green = 0.
            blue = (1. / 2. - x) / (1. / 2.)
        return red, green, blue
    def case7():
        # (1,0,0) -> (0.5,0.5,0.5) -> (0,0,1)
        #   red   ->    grey       ->  blue
        red = x
        blue = 1. - x
        green = 1. - x if x >= 0.5 else x
        return red, green, blue
    
    switch = {1:case1, 2:case2, 3:case3, 4:case4, 5:case5, 6:case6, 7:case7}
    func = switch[mod]
    return func()


if __name__ == "__main__":
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    import numpy as np


    def getposture():
        global EYE, LOOK_AT

        dist = np.sqrt(np.power((EYE - LOOK_AT), 2).sum())
        if dist > 0:
            phi = np.arcsin((EYE[1] - LOOK_AT[1]) / dist)
            theta = np.arcsin((EYE[0] - LOOK_AT[0]) / (dist * np.cos(phi)))
        else:
            phi = 0.0
            theta = 0.0

        return dist, phi, theta

    def init():
        glClearColor(0.0, 0.0, 0.0, 1.0)  # 设置画布背景色。注意：这里必须是4个参数
        glEnable(GL_DEPTH_TEST)  # 开启深度测试，实现遮挡关系
        glDepthFunc(GL_LEQUAL)  # 设置深度测试函数（GL_LEQUAL只是选项之一）


    def draw():
        global IS_PERSPECTIVE, VIEW
        global EYE, LOOK_AT, EYE_UP
        global SCALE_K
        global WIN_W, WIN_H

        # 清除屏幕及深度缓存
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 设置投影（透视投影）
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        if WIN_W > WIN_H:
            if IS_PERSPECTIVE:
                glFrustum(VIEW[0] * WIN_W / WIN_H, VIEW[1] * WIN_W / WIN_H, VIEW[2], VIEW[3], VIEW[4], VIEW[5])
            else:
                glOrtho(VIEW[0] * WIN_W / WIN_H, VIEW[1] * WIN_W / WIN_H, VIEW[2], VIEW[3], VIEW[4], VIEW[5])
        else:
            if IS_PERSPECTIVE:
                glFrustum(VIEW[0], VIEW[1], VIEW[2] * WIN_H / WIN_W, VIEW[3] * WIN_H / WIN_W, VIEW[4], VIEW[5])
            else:
                glOrtho(VIEW[0], VIEW[1], VIEW[2] * WIN_H / WIN_W, VIEW[3] * WIN_H / WIN_W, VIEW[4], VIEW[5])

        # 设置模型视图
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # 几何变换
        glScale(SCALE_K[0], SCALE_K[1], SCALE_K[2])

        # 设置视点
        gluLookAt(
            EYE[0], EYE[1], EYE[2],
            LOOK_AT[0], LOOK_AT[1], LOOK_AT[2],
            EYE_UP[0], EYE_UP[1], EYE_UP[2]
        )

        # 设置视口
        glViewport(0, 0, WIN_W, WIN_H)

        # --------------------------------------------------------------- draw the scaler bar

        width = 2.
        hight = 15.6

        ndy = 12
        dy = hight / ndy

        glBegin(GL_QUAD_STRIP)
        for ny in range(ndy):
            y0 = ny * dy
            r = y0 / hight
            red, green, blue = getColor(r, mod=5)
            glColor4f(red, green, blue, 1.0)
            glVertex2d(0., y0)
            glVertex2d(width, y0)

            y1 = y0 + dy
            r = y1 / hight
            red, green, blue = getColor(r, mod=5)
            glColor4f(red, green, blue, 1.0)
            glVertex2d(0., y1)
            glVertex2d(width, y1)

        glEnd()

        # ---------------------------------------------------------------
        glutSwapBuffers()  # 切换缓冲区，以显示绘制内容


    def reshape(width, height):
        global WIN_W, WIN_H

        WIN_W, WIN_H = width, height
        glutPostRedisplay()

    def mouseclick(button, state, x, y):
        global SCALE_K
        global LEFT_IS_DOWNED
        global MOUSE_X, MOUSE_Y

        MOUSE_X, MOUSE_Y = x, y
        if button == GLUT_LEFT_BUTTON:
            LEFT_IS_DOWNED = state == GLUT_DOWN
        elif button == 3:
            SCALE_K *= 1.05
            glutPostRedisplay()
        elif button == 4:
            SCALE_K *= 0.95
            glutPostRedisplay()

    def mousemotion(x, y):
        global LEFT_IS_DOWNED
        global EYE, EYE_UP
        global MOUSE_X, MOUSE_Y
        global DIST, PHI, THETA
        global WIN_W, WIN_H

        if LEFT_IS_DOWNED:
            dx = MOUSE_X - x
            dy = y - MOUSE_Y
            MOUSE_X, MOUSE_Y = x, y

            PHI += 2 * np.pi * dy / WIN_H
            PHI %= 2 * np.pi
            THETA += 2 * np.pi * dx / WIN_W
            THETA %= 2 * np.pi
            r = DIST * np.cos(PHI)

            EYE[1] = DIST * np.sin(PHI)
            EYE[0] = r * np.sin(THETA)
            EYE[2] = r * np.cos(THETA)

            if 0.5 * np.pi < PHI < 1.5 * np.pi:
                EYE_UP[1] = -1.0
            else:
                EYE_UP[1] = 1.0

            glutPostRedisplay()

    def keydown(key, x, y):
        global DIST, PHI, THETA
        global EYE, LOOK_AT, EYE_UP
        global IS_PERSPECTIVE, VIEW

        if key in [b'x', b'X', b'y', b'Y', b'z', b'Z']:
            if key == b'x':  # 瞄准参考点 x 减小
                LOOK_AT[0] -= 0.01
            elif key == b'X':  # 瞄准参考 x 增大
                LOOK_AT[0] += 0.01
            elif key == b'y':  # 瞄准参考点 y 减小
                LOOK_AT[1] -= 0.01
            elif key == b'Y':  # 瞄准参考点 y 增大
                LOOK_AT[1] += 0.01
            elif key == b'z':  # 瞄准参考点 z 减小
                LOOK_AT[2] -= 0.01
            elif key == b'Z':  # 瞄准参考点 z 增大
                LOOK_AT[2] += 0.01

            DIST, PHI, THETA = getposture()
            glutPostRedisplay()
        elif key == b'\r':  # 回车键，视点前进
            EYE = LOOK_AT + (EYE - LOOK_AT) * 0.9
            DIST, PHI, THETA = getposture()
            glutPostRedisplay()
        elif key == b'\x08':  # 退格键，视点后退
            EYE = LOOK_AT + (EYE - LOOK_AT) * 1.1
            DIST, PHI, THETA = getposture()
            glutPostRedisplay()
        elif key == b' ':  # 空格键，切换投影模式
            IS_PERSPECTIVE = not IS_PERSPECTIVE
            glutPostRedisplay()

    def cleanData(data, delimiter=' '):
        # make data to be the form that np.loadtxt can execute
        dt = []
        for i in range(len(data)):
            if data[i] == ',':
                dt.append(delimiter)
            elif data[i] == '\n':
                dt.append(delimiter)
            else:
                dt.append(data[i])
        dt = [''.join(dt)]
        return dt


    # -----------------------------------------------------------------------------------------------
    IS_PERSPECTIVE = True  # 透视投影
    VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 20.0])  # 视景体的left/right/bottom/top/near/far六个面
    SCALE_K = np.array([1.0, 1.0, 1.0])  # 模型缩放比例
    EYE = np.array([0.0, 0.0, 2.0])  # 眼睛的位置（默认z轴的正方向）
    LOOK_AT = np.array([0.0, 0.0, 0.0])  # 瞄准方向的参考点（默认在坐标原点）
    EYE_UP = np.array([0.0, 1.0, 0.0])  # 定义对观察者而言的上方（默认y轴的正方向）
    WIN_W, WIN_H = 640, 480  # 保存窗口宽度和高度的变量
    LEFT_IS_DOWNED = False  # 鼠标左键被按下
    MOUSE_X, MOUSE_Y = 0, 0  # 考察鼠标位移量时保存的起始位置

    DIST, PHI, THETA = getposture()  # 眼睛与观察目标之间的距离、仰角、方位角
    # -----------------------------------------------------------------------------------------------

    glutInit()
    displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH
    glutInitDisplayMode(displayMode)

    glutInitWindowSize(WIN_W, WIN_H)
    glutInitWindowPosition(300, 200)
    glutCreateWindow('Quidam Of OpenGL')

    init()  # 初始化画布
    glutDisplayFunc(draw)  # 注册回调函数draw()
    glutReshapeFunc(reshape)  # 注册响应窗口改变的函数reshape()
    glutMouseFunc(mouseclick)  # 注册响应鼠标点击的函数mouseclick()
    glutMotionFunc(mousemotion)  # 注册响应鼠标拖拽的函数mousemotion()
    glutKeyboardFunc(keydown)  # 注册键盘输入的函数keydown()

    glutMainLoop()  # 进入glut主循环