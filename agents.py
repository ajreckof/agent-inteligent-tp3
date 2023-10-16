import numpy as np

import warnings
import gymnasium as gym
from scipy.special import softmax
from random import choices
from IPython.display import clear_output
import utils


def floating_average(n):
    return lambda list : [ sum(list[max(0,i-n):i])/ min(i+1,n) for i in range(len(list))]

class BaseAgent():
	def __init__(self, env_name = "CartPole-v1", render_mode = None):
		"""Constructeur.
		
		Params
		======
			seed (int): random seed
		"""
		self.env_name = env_name
		self.env = gym.make(env_name, render_mode = render_mode)
		self.strategy = np.random.rand(self.env.action_space.n, len(self.env.observation_space.low))


	def getPolicy(self, state):
		return softmax(np.matmul(self.strategy,state))

	def getAction(self, state):
		policy = self.getPolicy(state)
		return choices(range(self.env.action_space.n), weights= policy, k = 1)[0]

	def saveStrategy(self, filename):
		np.savetxt("saved_strategies/" + filename, self.strategy)
	
	def loadStrategy(self, filename):
		self.strategy = np.loadtxt("saved_strategies/" + filename)

	def runOneEpisode(self):
		total_reward = 0
		self.observation, _ = self.env.reset()
		self.terminated = False
		self.truncated = False 
		while not (self.truncated or self.terminated):
			total_reward += self.runOneInteraction(self.observation)
		return total_reward

	def runOneInteraction(self, state):
		# agent policy that uses the observation and info
		action = self.getAction(state)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			self.observation, reward, self.terminated, self.truncated, _ = self.env.step(action)
		return reward 
	
	def test_agent(self, n_ep = 1):
		cumulative_reward = 0
		for _ in range(n_ep):
			cumulative_reward += self.runOneEpisode()
		return cumulative_reward / n_ep


	def noise_strategy(self, max_noise):
		noise = np.random.uniform(-max_noise, max_noise, self.strategy.shape)
		self.strategy = self.strategy + noise

	def change_render_mode(self,render_mode):
		self.env.close()
		self.env = gym.make(self.env_name, render_mode = render_mode)


class HillClimbingAgent(BaseAgent):
	def __init__(self, render_mode=None, env_name="CartPole-v1", noise = 3):
		super().__init__(render_mode, env_name)
		self.noise = noise

	def climb(self, stability_ep, max_ep , n_ep):
		reward_mem = []
		current_strategy_average_reward = self.test_agent(n_ep)
		best_strategy = self.strategy
		ep_since_last_best = 0
		for i in range(max_ep):
			if i % 10 == 0:
				clear_output(wait=True)
				print(i, " : ", current_strategy_average_reward)
				print(best_strategy)
			
			if ep_since_last_best == stability_ep:
				break

			self.noise_strategy(self.noise)
			reward = self.test_agent(n_ep)
			reward_mem.append(reward)

			if reward > current_strategy_average_reward :
				current_strategy_average_reward = reward
				best_strategy = self.strategy
				ep_since_last_best = 0
			else :
				self.strategy = best_strategy
				if reward == current_strategy_average_reward :
					current_strategy_average_reward = self.test_agent(n_ep)
				ep_since_last_best += 1
		
		utils.plot_sumrwd_mean_perepi(reward_mem,floating_average(max(10,i//10))(reward_mem))
		self.strategy = best_strategy
		return best_strategy, current_strategy_average_reward
		
class AdaptHillClimbingAgent(BaseAgent):
	def __init__(self, render_mode=None, env_name="CartPole-v1",  base_noise= 1):
		super().__init__(render_mode, env_name)
		self.base_noise = base_noise
		self.noise = base_noise

	def climb(self, max_noise = 10**3, noise_decay = 0.1, noise_increment= 1.03, max_ep= 1000 , n_ep=1):
		reward_mem = []
		current_strategy_average_reward = self.test_agent(n_ep)
		best_strategy = self.strategy
		ep_since_last_best = 0
		for i in range(max_ep):
			if i % 10 == 0:
				clear_output(wait=True)
				print("episodes : ", i)
				print("reward : ", current_strategy_average_reward)
				print("noise : ", self.noise)
				print(best_strategy)
			
			if self.noise > max_noise :
				break

			self.noise_strategy(self.noise)
			reward = self.test_agent(n_ep)
			reward_mem.append(reward)

			if reward > current_strategy_average_reward :
				current_strategy_average_reward = reward
				best_strategy = self.strategy
				self.noise = min(self.noise, self.base_noise) *noise_decay
			else:
				self.strategy = best_strategy
				self.noise = max(self.noise, self.base_noise)  * noise_increment
				if reward == current_strategy_average_reward :
					current_strategy_average_reward = self.test_agent(n_ep)
		
		utils.plot_sumrwd_mean_perepi(reward_mem,floating_average(max(10,i//10))(reward_mem))
		self.strategy = best_strategy
		return best_strategy, current_strategy_average_reward
		
		