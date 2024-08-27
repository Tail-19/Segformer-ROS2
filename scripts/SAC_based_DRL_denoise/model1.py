import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, nb_actions, nb_pstate, visual_length=2, freeze=False):
        super(QNetwork, self).__init__()

        self.encoder = Encoder(visual_length)
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        self.fc1 = nn.Linear(256+nb_pstate+nb_actions,128)
        self.fc2 = nn.Linear(128,32)
        self.fc3 = nn.Linear(32,nb_actions)
        
        self.fc11 = nn.Linear(256+nb_pstate+nb_actions,128)
        self.fc21 = nn.Linear(128,32)
        self.fc31 = nn.Linear(32,nb_actions)

        self.apply(weights_init_)

    def forward(self, inp):
        istate, pstate, a = inp

        x1 = istate
        x1 = self.avg(self.encoder(x1))
        x1 = x1.view(x1.size(0), -1)
        
        x2 = pstate
        
        x = torch.cat([x1, x2, a], dim=1)
        
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc11(x))
        q2 = F.relu(self.fc21(q2))
        q2 = self.fc31(q2)

        return q1, q2


class GaussianPolicy(nn.Module):
    def __init__(self, nb_actions, nb_pstate, visual_length=2, freeze=False, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.encoder = Encoder(visual_length)
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1,1))
                
        self.fc1 = nn.Linear(256+nb_pstate,128)
        self.fc2 = nn.Linear(128,32)

        self.mean_linear = nn.Linear(32, nb_actions)
        self.log_std_linear = nn.Linear(32, nb_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, inp):
        istate, pstate = inp
        x1 = istate

        x1 = self.avg(self.encoder(x1))
        x1 = x1.view(x1.size(0), -1)

        x2 = pstate
        
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, inp):
        mean, log_std = self.forward(inp)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, std

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, nb_actions, nb_pstate, visual_length=2, freeze=False, action_space=None):
        super(DeterministicPolicy, self).__init__()
        
        self.encoder = Encoder(visual_length)
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1,1))
                
        self.fc1 = nn.Linear(256+nb_pstate,128)
        self.fc2 = nn.Linear(128,32)
        self.mean = nn.Linear(32, nb_actions)
        self.noise = torch.Tensor(nb_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, inp):
        istate, pstate = inp
        x1 = istate

        x1 = self.avg(self.encoder(x1))
        x1 = x1.view(x1.size(0), -1)

        x2 = pstate
        
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, inp):
        mean = self.forward(inp)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)





class Encoder(nn.Module):
    def __init__(self, visual_length=2):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(visual_length,16,5, stride=2)
        self.conv2 = nn.Conv2d(16,64,5, stride=2)
        self.conv3 = nn.Conv2d(64,256,5, stride=2)
        
        
    def forward(self, inp):
        x1 = inp

        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))

        return x1



class Decoder(nn.Module):
    def __init__(self, visual_length=2):
        super(Decoder, self).__init__()
        
        self.deco1 = nn.ConvTranspose2d(256,64,6, stride=2)
        self.deco2 = nn.ConvTranspose2d(64,16,6, stride=2)
        self.deco3 = nn.ConvTranspose2d(16,visual_length,6, padding=(1,0), stride=2)
        
    def forward(self, latent):
        y1 = F.relu(self.deco1(latent))
        y1 = F.relu(self.deco2(y1))
        y1 = F.relu(self.deco3(y1))
        
        return y1



class Autoencoder(nn.Module):
    def __init__(self, visual_length=2):
        super(Autoencoder, self).__init__()
        
        self.encoder = Encoder(visual_length)
        self.decoder = Decoder(visual_length)
        
    def forward(self, inp):
        
        x = self.encoder(inp)
        y = self.decoder(x)
        
        return y
    
    def save_encoder(self, output='.'):
        torch.save(self.encoder.state_dict(), '{}/encoder.pkl'.format(output))
    
    def save_decoder(self, output='.'):
        torch.save(self.decoder.state_dict(), '{}/decoder.pkl'.format(output))
    
    def load_encoder(self, output):
        if output is None: return
        self.encoder.load_state_dict(torch.load('{}/encoder.pkl'.format(output)))
    
    def load_decoder(self, output):
        if output is None: return
        self.decoder.load_state_dict(torch.load('{}/decoder.pkl'.format(output)))