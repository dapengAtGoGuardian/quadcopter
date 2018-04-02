import numpy as np
from physics_sim import PhysicsSim

class Task():
    def __init__(self, target_pos=None,
                 init_pose=None, init_velocities=None, init_angle_velocities=None, runtime=128):
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_state(self):
        return np.concatenate((self.sim.pose, self.sim.v, self.sim.angular_v))

    def get_reward(self):
        '''
        only consider the distance to the target; hopefully other factors are taken care of by actions
        '''
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum() # this only considers positions; but not rotation...
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum() - .1*abs(self.sim.pose[3:6]).sum()# don't rotate!
        return reward

    def step(self, rotor_speeds):
        ''' # not sure why we need to repeat the action -- keep the code here
        reward = 0
        pose_all = []
        for _ in range(1):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        '''
        done = self.sim.next_timestep(rotor_speeds)
        reward = self.get_reward() 
        next_state = self.get_state()
        return next_state, reward, done

    def reset(self):
        self.sim.reset()
        return self.get_state()
