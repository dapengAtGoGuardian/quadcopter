import numpy as np
from physics_sim import PhysicsSim

class Task():
    '''
    take off from default (0, 0, 10) to surpass a higher z, default (0, 0, 25)
    '''
    def __init__(self, target_z=25, init_z=10, init_z_velocity=0, runtime=512):
        self.target_z = target_z
        init_pose, init_velocities, init_angle_velocities = np.zeros(6), np.zeros(3), np.zeros(3)
        init_pose[2] = init_z
        init_velocities[2] = init_z_velocity
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.stepCount = 0
        self.max_runtime = runtime

    # [z_pos, z_speed]
    def get_state(self):
        return np.array([self.sim.pose[2], self.sim.v[2]])

    '''
    this is a popular reward function
    '''
    def step(self, rotor_speed):
        self.stepCount += 1
        self.sim.next_timestep(np.repeat(rotor_speed, 4))
        next_state = self.get_state()
        reward = -min(abs(self.target_z - self.sim.pose[2]), 20.0)
        done = False
        if self.sim.pose[2] >= self.target_z:  # agent has crossed the target height
            reward += 0.0  # bonus reward
            done = True
        elif self.stepCount > self.max_runtime:  # agent has run out of time
            reward -= 10.0  # extra penalty
            done = True
        elif self.sim.pose[2] ==0 and self.sim.v[2] < 0:
            reward -= 10.0
            done = True
        return next_state, reward, done

    '''
    this is great help! though agent as quickly found a solution, 
    later it got lost -- couldn't surpass target_z-- with 0.8 gamma; or stop at 87 steps with 0.99 gamma
    '''
    def step_1(self, rotor_speed):
        self.stepCount += 1
        done = False
        z0 = self.sim.pose[2]
        prevDist = np.abs(z0 - self.target_z)
        self.sim.next_timestep(np.repeat(rotor_speed, 4))
        next_state = self.get_state()
        z1 = self.sim.pose[2]
        currDist = np.abs(z1 - self.target_z)
        reward = 10 * (prevDist - currDist)
        if self.sim.pose[2] >= self.target_z:  # agent has crossed the target height
            done = True
        if self.stepCount > self.max_runtime:  # agent has run out of time
            reward -= 10.0  # extra penalty
            done = True
        elif self.sim.pose[2] ==0 and self.sim.v[2] < 0:
            reward -= 10.0
            done = True
        return next_state, reward, done

    def reset(self):
        self.sim.reset()
        self.stepCount = 0
        return self.get_state()

