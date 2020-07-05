import gym
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import count
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage
import math
import random
from Net import DQNet
from ReplayMemory import ReplayMemory, Transition
import torch.nn.functional as F
import PIL.Image as Image

class Assault():

    def __init__(self, env_name):
        super(Assault).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Working on " + torch.cuda.get_device_name(self.device))
        self.env = gym.make(env_name)
        self.n_actions = self.env.action_space.n
        self.eps_start, self.eps_end, self.eps_decay = 1, 0.02, 50000
        self.gamma = 0.99
        self.policy_update = 1
        self.target_update = 1000
        self.steps_done = 0

        self.resize = Compose([ToPILImage(), Resize(60, interpolation=Image.CUBIC), ToTensor()])

        self.policy_net = DQNet(self.get_state(self.env.reset(), 'cpu').unsqueeze(0), self.n_actions).to(self.device)
        self.target_net = DQNet(self.get_state(self.env.reset(), 'cpu').unsqueeze(0), self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)

        self.memory = ReplayMemory(10000)
        self.batch_size = 32

        self.env.reset()


    def run(self, episodes):

        plt.ion()
        plt.show()

        self.losses_hist = []
        total_rewards_hist = []

        for ep in range(episodes):
            state = self.get_state(self.env.reset())
            next_state = self.get_state(self.env.reset())

            total_rewards = 0
            self.losses = []

            for i in count():
                if ep%50 == 0:
                    self.env.render()

                state = next_state

                action = self.select_action(state)
                next_obs, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                total_rewards += reward.item()

                next_state = self.get_state(next_obs)

                self.memory.push(state, action, next_state, reward)

                if i % self.policy_update == 0:
                    self.optimize_model()

                if done:
                    #self.scheduler.step()
                    break

                if self.steps_done % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())


            total_rewards_hist.append(total_rewards)
            self.losses_hist.append(np.mean(self.losses))
            #plt.plot(total_rewards_hist)
            plt.plot(self.losses_hist)
            plt.draw()
            plt.pause(0.001)
            print("Episode", ep, ":", total_rewards)

        self.env.close()
        print("Complete")

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.cat(batch.reward)

        #print("ONE")
        #print(state_batch.shape)
        #print(action_batch)
        #print(reward_batch)
        #print(non_final_next_states.shape)

        state_actions_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        #print(next_state_values)
        expected_values = next_state_values * self.gamma + reward_batch
        #print(expected_values)

        #print("Flag 1")
        #print(state_actions_values)
        #print(expected_values.unsqueeze(1))
        loss = F.smooth_l1_loss(state_actions_values, expected_values.unsqueeze(1))
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start-self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if self.steps_done % 5000 == 0:
            print("step:", self.steps_done, "/ eps:", eps_threshold)

        if sample > eps_threshold:
            return self.policy_net(state.unsqueeze(0)).max(1)[1]
        else:
            return torch.tensor([random.randrange(self.n_actions)], device=self.device, dtype=torch.long)


    def get_state(self, obs, device=None):
        if device is None:
            device = self.device

        coeff_up = 0.2
        coeff_down = 0.1

        h, w, c = obs.shape
        new_top = int(h * coeff_up)
        new_down = int(h - h * coeff_down)

        state = obs[new_top:new_down, : , :]

        return self.resize(state).to(device)



if __name__ == '__main__':
    env_name = "Assault-v0"
    a = Assault(env_name)
    a.run(500)