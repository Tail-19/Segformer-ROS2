import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from SAC_based_DRL_denoise.utils import soft_update, hard_update
from SAC_based_DRL_denoise.model import GaussianPolicy, QNetwork, DeterministicPolicy, Autoencoder
from torch.distributions import Normal
from cpprb import PrioritizedReplayBuffer



class DRL(object):
    def __init__(self, action_dim, pstate_dim, VISUAL_LENGTH = 3, policy = 'Gaussian', 
                 LR_C = 3e-4, LR_A = 3e-4, BUFFER_SIZE=1e3, HUMAN = 'True',
                 LOAD_ENCODER='True', FREEZE_ENCODER='True',
                 TAU=5e-3, POLICY_FREQ = 2, GAMMA = 0.99, ALPHA=0.05, automatic_entropy_tuning=True):
        
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
                                                   "done": {}},
                                                  next_of=("obs","pobs"), stack_compress="obs",
                                                  default_dtype=np.float16)

        self.critic = QNetwork(self.action_dim, self.pstate_dim, self.visual_length, FREEZE_ENCODER).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), LR_C)

        self.critic_target = QNetwork(self.action_dim, self.pstate_dim, self.visual_length, FREEZE_ENCODER).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -(self.action_dim)
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=LR_A)

            self.policy = GaussianPolicy(self.action_dim, self.pstate_dim, self.visual_length, FREEZE_ENCODER).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=LR_A)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(self.action_dim, self.pstate_dim, self.visual_length, FREEZE_ENCODER).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=LR_A)
        
        if LOAD_ENCODER:
            self.load_encoder('/home/yanxin/phd_workspace/e2e_nav/scripts/SAC_based_DRL_denoise/')
            print('Encoder loaded! Encoder weights freezed: {}!'.format(FREEZE_ENCODER))
        
        
    def choose_action(self, istate, pstate, evaluate=False):
        istate = torch.FloatTensor(istate).float().unsqueeze(0).permute(0,3,1,2).to(self.device)
        pstate = torch.FloatTensor(pstate).float().unsqueeze(0).to(self.device)
        
        if evaluate is False:
            action, _, _, _ = self.policy.sample([istate, pstate])
        else:
            _, _, action, _ = self.policy.sample([istate, pstate])
        return action.detach().squeeze(0).cpu().numpy()

    def learn(self, batch_size=64):
        # Sample a batch from memory
        data = self.replay_buffer.sample(batch_size)
        
        istates, pstates = data['obs'], data['pobs']
        actions, actions_exp, interv =  data['act'], data['acte'], data['intervene']
        rewards, next_istates, next_pstates, dones = data['rew'], data['next_obs'], data['next_pobs'], data['done']
        indexes, weights = data['indexes'], data['weights']
        
        istates = torch.FloatTensor(istates).permute(0,3,1,2).to(self.device)
        pstates = torch.FloatTensor(pstates).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        actions_exp = torch.FloatTensor(actions_exp).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_istates = torch.FloatTensor(next_istates).permute(0,3,1,2).to(self.device)
        next_pstates = torch.FloatTensor(next_pstates).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _, _ = self.policy.sample([next_istates, next_pstates])
            qf1_next_target, qf2_next_target = self.critic_target([next_istates, next_pstates, next_state_actions])
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic([istates, pstates, actions])  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        
        td_errors = (abs(qf1.detach() - next_q_value))[:,0]
        
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, means, stds = self.policy.sample([istates, pstates])
        
        qf1_pi, qf2_pi = self.critic([istates, pstates, pi])
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        if self.human_guidance:
            index_imi, _ = np.where(interv==1)
            actions_imi = actions[index_imi]
            means = means[index_imi]
            stds = stds[index_imi]
            # self.dist = Normal(means, stds)
            
            if len(index_imi) > 0:
                # imitation_loss = -self.dist.log_prob(actions_imi).mean()
                imitation_loss = F.gaussian_nll_loss(means, actions_imi, stds**2)
                # print('imitation loss is {} with amount {}'.format(imitation_loss.copy().detach().cpu().numpy(),len(actions_imi)))
                
                q_adv = (torch.exp(self.critic([istates, pstates, actions])[0][:,0] - qf1_pi[:,0])).detach()
                q_weight = torch.zeros_like(q_adv)
                q_weight[index_imi] = 1
                qa_errors = q_adv*q_weight

            else:
                imitation_loss = 0.
                qa_errors = 0.
                
            policy_loss = ((self.alpha * log_pi) - min_qf_pi + imitation_loss).mean()
        else:
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            qa_errors = 0.
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if self.itera % self.policy_freq == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        self.itera += 1
        
        priorities = td_errors + qa_errors
        priorities = priorities.cpu().numpy()
        
        self.replay_buffer.update_priorities(indexes, priorities)

        return qf1_loss.item(), policy_loss.item()
    
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
        self.critic_target.load_state_dict(torch.load('{}/critic.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))

    def save_model(self, output):
        torch.save(self.policy.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))

    def save_transition(self, output, timeend=0):
        self.replay_buffer.save_transitions(file='{}/{}'.format(output, timeend))

    def load_transition(self, output):
        if output is None: return
        self.replay_buffer.load_transitions('{}.npz'.format(output))
    
    def load_encoder(self, output):
        if output is None: return
        self.policy.encoder.load_state_dict(torch.load('{}/encoder.pkl'.format(output), map_location=torch.device('cpu')))
        self.critic.encoder.load_state_dict(torch.load('{}/encoder.pkl'.format(output), map_location=torch.device('cpu')))
        self.critic_target.encoder.load_state_dict(torch.load('{}/encoder.pkl'.format(output), map_location=torch.device('cpu')))
