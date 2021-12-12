from collections import defaultdict
from typing import DefaultDict
from PIDController import PIDController
import matplotlib.pyplot as plt
from Controller import Controller
from PIDController import PIDController
import numpy as np

# controller = Controller(["/home/ramos/phiflow/neural_control/controller_design/controller_coefficients1.json", "/home/ramos/phiflow/neural_control/controller_design/controller_coefficients2.json"])
# controller = Controller(["/home/ramos/phiflow/neural_control/controller_design/ls_coeffs_complex1.json", "/home/ramos/phiflow/neural_control/controller_design/ls_coeffs_complex2.json"])
# controller = PIDController(["/home/ramos/phiflow/neural_control/controller_design/gains_pid_simple.json"], 0, [20])
controller = PIDController(["/home/ramos/phiflow/neural_control/controller_design/gains_pid_box_xy.json", "/home/ramos/phiflow/neural_control/controller_design/gains_pid_box_angle.json"], 0, [20, 0.5])
controller.reset()
obj = np.array((20, 20))
# obj = np.array((86 - 85, 56 - 55))
obj2 = np.array((3,))
# 10.092369, 21.150137  -2.13455350626117
# obj = np.array((10.0,))
x0 = 0
nt = 1000
mass = 36
mass2 = 4000
dt = controller.dt
x = x0
x2 = x0
vel = 0
vel2 = 0
storage = defaultdict(list)
for n in range(nt):
    if n > 2500:
        # obj = np.array((130, 80))
        #     obj = np.array((2))
        pass
    error = obj - x
    error2 = obj2 - x2
    effort = controller([error, error2])
    # print(controller.integrator)
    # effort = controller([error])
    # t = 0
    u = effort[0]
    t = effort[1]
    # u = u + 10
    acc = u / mass
    vel = vel + acc * dt
    x = x + vel * dt

    acc2 = t / mass2
    vel2 = vel2 + acc2 * dt
    x2 = x2 + vel2 * dt

    storage['error_xy'].append(error)
    storage['error_angle'].append(error2)
    storage['force_xy'].append(u)
    storage['torque_angle'].append(t)
    storage['vel_xy'].append(vel)
    storage['vel_angle'].append(vel2)
    storage['integrator'].append(controller.integrator[0][0])

storage = {key: np.array(value) for key, value in storage.items()}


fig, axes = plt.subplots(3, 1)
axes[0].plot(storage['error_xy'])
axes[0].set_title('error_xy')
axes[1].plot(storage['force_xy'])
axes[1].set_title('force_xy')
axes[2].plot(storage['vel_xy'])
axes[2].set_title('vel_xy')
for axes in axes:
    axes.grid()
fig.savefig('debug1.png')

fig, axes = plt.subplots(3, 1)
axes[0].plot(storage['error_angle'])
axes[0].set_title('error_angle')
axes[1].plot(storage['torque_angle'])
axes[1].set_title('torque_angle')
axes[2].plot(storage['vel_angle'])
axes[2].set_title('vel_angle')
for axes in axes:
    axes.grid()
fig.savefig('debug2.png')

fig = plt.figure()
plt.plot(storage['integrator'])
fig.savefig('debug23.png')
