import numpy as np
import pygame
import signal
import matplotlib.pyplot as plt
import random
import torch
import time

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



from SAC_based_DRL_denoise.autoencoder import Encode

VISUAL_LENGTH = 3

agent = Encode(action_dim=2, pstate_dim=3, VISUAL_LENGTH=VISUAL_LENGTH, LR=3e-4, BUFFER_SIZE=1e1)
agent.model.load_encoder('.')
agent.model.load_decoder('.')
# agent.policy.load_state_dict(torch.load('actor.pkl'))
# agent.load_transition(0)
# agent.load_model('.')

from env_design_dynamic import CarlaEnv

env = CarlaEnv(state_width=180, state_height=90, world=1, surrounding_number_vehicle=30,
               surrounding_number_walker=30)

MAX_EPISODES = 10

real_total_step = 0
total_step = 0
loss = []


for i in range(MAX_EPISODES):
    step = 0
    previous_steer = []
    previous_pedal = []
    
    observation, p = env.reset()
    observation_real = env.get_raw_image()
    istate = np.repeat(np.expand_dims(observation,2), VISUAL_LENGTH, axis=2)
    istate1 = istate.copy()
    istate2 = np.repeat(np.expand_dims(observation_real,2), VISUAL_LENGTH, axis=2)
    pstate = np.array([0.,0.,0.])
    done = False
    
    while not done:
        action = np.random.uniform(-0.2,1), np.random.uniform(-1,1)

        observation_, reward, done, p, human_action = env.step(action)
        
        observation_real_ = env.get_raw_image()
        

        for j in range(VISUAL_LENGTH-1):
            istate1[:,:,j] = istate1[:,:,j+1].copy()
            istate1[:,:,VISUAL_LENGTH-1] = observation_.copy()
        
        for j in range(VISUAL_LENGTH-1):
            istate2[:,:,j] = istate2[:,:,j+1].copy()
            istate2[:,:,VISUAL_LENGTH-1] = observation_real_.copy()

        if step%50==0:
            plt.subplot(121)
            plt.imshow(istate)
            plt.subplot(122)
            plt.imshow(agent.forward_encoder(istate))
            plt.show()
        
        observation = observation_.copy()
        istate = istate1.copy()
        step += 1
        total_step += 1
        real_total_step += 1
        

        
        signal.signal(signal.SIGINT, env.signal_handler)
    

    print('Episode: {}'.format(i))
    print('Total step: {}\n'.format(total_step))
    

pygame.display.quit()
pygame.quit()



