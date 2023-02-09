import roboticstoolbox as rtb
import numpy as np
import spatialmath as sm
import spatialgeometry as sg
from roboticstoolbox.backends.swift import Swift   # initiate 3D browser viewer

# init swift environment
env = Swift()
env.launch(realtime=True)

# load robot
robot = rtb.models.URDF.UR5()
robot.q = robot.q1

# add robot to swift
env.add(robot)

for x in np.linspace(-0.01, 0.01, 100):
      # goal position, offset by args in m
      # fkine finds end effector position (forward kinematics)
      Tep =  robot.fkine(robot.q) * sm.SE3.Tx(x)

      # axes
      axes = sg.Axes(length=0.1, base=Tep)
      env.add(axes)

      arrived = False      # arrived flag

      dt = 0.01
      while not arrived:

            # v is vector representing spatial error
            # gain is speed to goal, thresh is when arrived=True
            v, arrived = rtb.p_servo(robot.fkine(robot.q), Tep, gain=50, threshold=0.01)

            # Jacobian of robot in end effector frame
            J = robot.jacobe(robot.q)

            # desired joint velocity of robot, qd is joint velocity in rad
            robot.qd = np.linalg.pinv(J) @ v

            # step the env
            print(robot.q)
            env.step(dt)

# stop browser tab from closing
env.hold()
