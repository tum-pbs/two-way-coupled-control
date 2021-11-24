from PIDController import PIDController
import matplotlib.pyplot as plt
from Controller import Controller
from PIDController import PIDController
import numpy as np

# controller = Controller(["/home/ramos/phiflow/neural_control/controller_design/controller_coefficients1.json", "/home/ramos/phiflow/neural_control/controller_design/controller_coefficients2.json"])
# controller = Controller(["/home/ramos/phiflow/neural_control/controller_coefficients.json", "/home/ramos/phiflow/neural_control/controller_coefficients2.json"])
# controller = PIDController(["/home/ramos/phiflow/neural_control/controller_design/gains_pid1.json"], 0, [10])
controller = PIDController(["/home/ramos/phiflow/neural_control/controller_design/gains_pid1.json", "/home/ramos/phiflow/neural_control/controller_design/gains_pid2.json"], 0, [10, 0.1])
controller.reset()
obj = np.array((10, 15))
# obj = np.array((86 - 85, 56 - 55))
obj2 = np.array((2,))
# 10.092369, 21.150137  -2.13455350626117
# obj = np.array((10.0,))
x0 = 0
nt = 6000
mass = 180
mass2 = 6350
dt = controller.dt
x = x0
x2 = x0
vel = 0
vel2 = 0
storage = np.zeros((nt, 7))
# p = np.zeros((nt, 2))
# i = np.zeros((nt, 2))
# d = np.zeros((nt, 2))
for n in range(nt):
    if n > 2500:
        obj = np.array((130, 80))
        #     obj = np.array((2))
    error = obj - x
    error2 = obj2 - x2
    effort = controller([error, error2])
    # effort = controller([error])
    # print(effort)
    u = effort[0]
    t = effort[1]
    # u = u + 10
    acc = u / mass
    vel = vel + acc * dt
    x = x + vel * dt

    acc2 = t / mass2
    vel2 = vel2 + acc2 * dt
    x2 = x2 + vel2 * dt

    storage[n, :2] = u
    storage[n, 2:4] = vel2
    storage[n, 4:6] = error
    storage[n, 6] = x2

storage = np.array(storage)

# Load test
# error_x = np.array([np.load(f'/home/ramos/phiflow/storage/controller/dummy/tests/test1/data/error_x/error_x_case0000_{i:05d}.npy') for i in range(nt)])
# error_y = np.array([np.load(f'/home/ramos/phiflow/storage/controller/dummy/tests/test1/data/error_y/error_y_case0000_{i:05d}.npy') for i in range(nt)])
# control_force_x = np.array([np.load(f'/home/ramos/phiflow/storage/controller/dummy/tests/test1/data/control_force_x/control_force_x_case0000_{i:05d}.npy') for i in range(nt)])
# control_force_y = np.array([np.load(f'/home/ramos/phiflow/storage/controller/dummy/tests/test1/data/control_force_y/control_force_y_case0000_{i:05d}.npy') for i in range(nt)])

plt.figure()
# plt.plot(np.arange(nt) * dt, storage[:, 0], label='model_x')
# plt.plot(np.arange(nt) * dt, storage[:, 1], label='model_y')
plt.plot(np.arange(nt) * dt, storage[:, 2], 'x', label='model_x')
plt.plot(np.arange(nt) * dt, storage[:, 3], 'x', label='model_y')
# plt.plot(np.arange(nt) * dt, error_x, label='fluidsim_x')
# plt.plot(np.arange(nt) * dt, error_y, label='fluidsim_y')

# plt.plot(np.arange(nt) * dt, control_force_x, '+', label='fluidsim_x')
# plt.plot(np.arange(nt) * dt, control_force_y, '+', label='fluidsim_y')

plt.ylabel('Control Force')
plt.legend()

plt.figure()
plt.plot(np.arange(nt) * dt, storage[:, 4])
plt.plot(np.arange(nt) * dt, storage[:, 5])

# plt.figure()
# plt.plot(np.arange(nt) * dt, storage[:, 4])
# plt.plot(np.arange(nt) * dt, storage[:, 5])

# plt.figure()
# plt.plot(np.arange(nt) * dt, storage[:, 6])


# plt.figure()
# plt.title('Gains for x')
# plt.plot(np.arange(nt) * dt, p[:, 0], label='P')
# plt.plot(np.arange(nt) * dt, i[:, 0], label='I')
# plt.plot(np.arange(nt) * dt, d[:, 0], label='D')
# plt.legend()

# plt.figure()
# plt.title('Gains for y')
# plt.plot(np.arange(nt) * dt, p[:, 1], label='P')
# plt.plot(np.arange(nt) * dt, i[:, 1], label='I')
# plt.plot(np.arange(nt) * dt, d[:, 1], label='D')
# plt.legend()
plt.show()
