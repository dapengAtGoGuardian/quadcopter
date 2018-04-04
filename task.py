import numpy as np
from physics_sim import PhysicsSim

class Task():
    def __init__(self, target_pos=None,
                 init_pose=None, init_velocities=None, init_angle_velocities=None, runtime=128):
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_state(self):
        return np.concatenate((self.sim.pose, self.sim.v, self.sim.angular_v))

    '''
    def get_reward(self):
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum() # this only considers positions; but not rotation...
        reward = - .1*abs(self.sim.pose[3:6]).sum()# don't rotate!
        return reward
    '''

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
        prevState = self.get_state()
        prev_angular_v = np.copy(self.sim.angular_v)
        done = self.sim.next_timestep(rotor_speeds)
        currentState = self.get_state()
        #reward = self.get_reward() 
        #np.square(prevState[:3] - self.target_pos).sum()
        # improvement of distance + no rotation + no offset
        #reward =  1*(np.sqrt(np.square(prevState[:3] - self.target_pos).sum()) -\
        #    np.sqrt(np.square(currentState[:3] - self.target_pos).sum())) -\
        reward_distance = np.sqrt(np.square(prevState[:3] - self.target_pos).sum()) -\
            np.sqrt(np.square(currentState[:3] - self.target_pos).sum())
        reward_rotation = np.abs(prevState[3:6]).sum() - np.abs(currentState[3:6]).sum()
        reward_xy_offset = np.abs(prevState[:2]).sum() - np.abs(currentState[:2]).sum()
        reward = 3 * reward_distance + reward_rotation + reward_xy_offset
        '''
        reward = 1 - np.abs(self.target_pos[2] - currentState[2]) -\
            abs(self.sim.pose[3:6]).sum() -\
            abs(self.sim.pose[:2]).sum()
        '''
        next_state = self.get_state()
        return next_state, reward, done

    def reset(self):
        self.sim.reset()
        return self.get_state()
