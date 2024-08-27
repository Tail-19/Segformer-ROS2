import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from SAC_based_DRL.utils import soft_update, hard_update
from SAC_based_DRL.model_large import GaussianPolicy, QNetwork, DeterministicPolicy

from cpprb import PrioritizedReplayBuffer

from torch.distributions import Normal

class DRL(object):
    def __init__(self, action_dim, pstate_dim, VISUAL_LENGTH = 2, policy = 'Gaussian',
                 LR_C = 3e-4, LR_A = 3e-4, BUFFER_SIZE=5e4, HUMAN = 'False',
                 TAU=5e-3, POLICY_FREQ = 2, GAMMA = 0.99, ALPHA=0.05,
                 automatic_entropy_tuning=True):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.gamma = GAMMA
        self.tau = TAU
        self.alpha = ALPHA
        self.human_guidance = HUMAN

        self.pstate_dim = pstate_dim
        self.action_dim = action_dim
        
        self.itera = 0

        self.policy_type = policy
        self.policy_freq = POLICY_FREQ
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        self.nllloss = nn.NLLLoss()
        
        self.visual_length = VISUAL_LENGTH
        
        self.replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE,
                                                  {"obs": {"shape": (90,180,VISUAL_LENGTH)},
                                                   "pobs": {"shape":pstate_dim},
                                                   "act": {"shape":action_dim},
                                                   "acte": {"shape":action_dim},
                                                   "intervene": {},
                                                   "rew": {},
                                                   "next_obs": {"shape": (90,180,VISUAL_LENGTH)},
                                                   "next_pobs": {"shape":pstate_dim},
                                                   "done": {}},
                                                  next_of=("obs"))
        self.critic = QNetwork(self.action_dim, self.pstate_dim, self.visual_length).to(device=self.device)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -(self.action_dim)
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=LR_A)

            self.policy = GaussianPolicy(self.action_dim, self.pstate_dim, self.visual_length).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=LR_A)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(self.action_dim, self.pstate_dim, self.visual_length).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=LR_A)

    def choose_action(self, istate, pstate, evaluate=False):
        istate = torch.FloatTensor(istate).float().unsqueeze(0).permute(0,3,1,2).to(self.device)
        pstate = torch.FloatTensor(pstate).float().unsqueeze(0).to(self.device)
        
        if evaluate is False:
            action, _, _ = self.policy.sample([istate, pstate])
        else:
            _, _, action = self.policy.sample([istate, pstate])
        return action.detach().squeeze(0).cpu().numpy()

    def learn(self, batch_size=64):
        # Sample a batch from memory
        data = self.replay_buffer.sample(batch_size)
        istates, pstates = data['obs'], data['pobs']
        actions =  data['act']
        istates = torch.FloatTensor(istates).permute(0,3,1,2).to(self.device)
        pstates = torch.FloatTensor(pstates).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)


        pi, log_pi, mean = self.policy.sample([istates, pstates])
        
        mean, log_std = self.policy.forward([istates, pstates])
        self.dist = Normal(mean, log_std.exp())

        policy_loss = -self.dist.log_prob(actions).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.itera += 1
        
        return policy_loss.item(), policy_loss.item()
    
    # Define the storing by priority experience replay
    def store_transition(self,  s, ps, a, ae, i, r, s_, ps_, d=0):
        self.replay_buffer.add(obs=s,
                               pobs=ps,
                               act=a,
                               acte=ae,
                               intervene=i,
                               rew=r,
                               next_obs=s_,
                               next_pobs=ps_,
                               done=d)
    

    # Save and load model parameters
    def load_model(self, output):
        if output is None: return
        self.policy.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))

    def save_model(self, output):
        torch.save(self.policy.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))

    def save_transition(self, output, timeend=0):
        self.replay_buffer.save_transitions(file='{}/{}'.format(output, timeend))

    def load_transition(self, output):
        if output is None: return
        self.replay_buffer.load_transitions('{}.npz'.format(output))

