"""
This module contains the SRSWTI environment and RL Orchestrator for training and prediction.
"""

import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from advanced_state import AdvancedState

class SRSWTIEnv(gym.Env):
    """
    SRSWTI (Scientific Research and Scholarly Writing Thought Instigator) Environment
    """

    def __init__(self):
        super().__init__()

        self.state = AdvancedState()

        # Define flattened action space
        self.action_space = spaces.MultiDiscrete([3, 10, 5, 5, 5, 3, 5])

        # Define observation space
        primary_dim = self.state.primary_state.shape[0]
        secondary_dim = 7  # Adjust this based on the number of secondary state variables
        self.observation_space = spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(primary_dim + secondary_dim,),
            dtype=np.float32
        )

    def _get_obs(self):
        primary, secondary = self.state.get_state()
        secondary_obs = np.array([
            len(secondary['user_history']),
            secondary['resource_utilization'],
            len(secondary['pfc_insights']),
            len(secondary['domain_knowledge']),
            len(secondary['search_history']),
            secondary['session_duration'],
            secondary['breakthrough_potential']
        ])
        return np.concatenate([primary, secondary_obs])

    def reset(self):
        self.state = AdvancedState()
        return self._get_obs()

    def step(self, action):
        # Unpack action
        engine, fe_ie_balance, fe_worker, ie_worker, pfc_worker, query_refinement, shot_count = action

        # Convert to dictionary for easier handling
        action_dict = {
            'engine': engine,
            'fe_ie_balance': fe_ie_balance,
            'fe_worker': fe_worker,
            'ie_worker': ie_worker,
            'pfc_worker': pfc_worker,
            'query_refinement': query_refinement,
            'shot_count': shot_count
        }

        # Simulate action effects
        self._simulate_action_effects(action_dict)

        # Calculate reward
        reward = self._calculate_reward(action_dict)

        # Check if episode is done
        done = self.state.primary_state[self.state.embedding_dim + 1] >= 1.0  # Research progress

        return self._get_obs(), reward, done, {}

    def _simulate_action_effects(self, action):
        """
        Simulate the effects of the chosen action on the environment state.

        Args:
            action (dict): A dictionary containing the action parameters.
        """
        engine = action['engine']
        fe_ie_balance = action['fe_ie_balance'] / 9.0  # Normalize to 0-1 range
        
        # Update research progress
        if engine == 0:  # FE
            progress_increase = 0.05 * (1 - fe_ie_balance)
        elif engine == 1:  # IE
            progress_increase = 0.05 * fe_ie_balance
        else:  # PFC
            progress_increase = 0.03
        
        new_progress = self.state.primary_state[self.state.embedding_dim + 1] + progress_increase
        self.state.update_research_progress(new_progress)
        
        # Update user engagement
        new_engagement = self.state.primary_state[-2] + np.random.normal(0, 0.1)
        self.state.update_user_engagement(np.clip(new_engagement, 0, 1))
        
        # Update query complexity based on query refinement
        if action['query_refinement'] == 1:  # Expand
            new_complexity = self.state.primary_state[-1] + 0.1
            self.state.update_query_complexity(min(1, new_complexity))
        elif action['query_refinement'] == 2:  # Narrow
            new_complexity = self.state.primary_state[-1] - 0.1
            self.state.update_query_complexity(max(0, new_complexity))
        
        # Update shot count and efficiency
        new_shot_count = self.state.primary_state[-5] + action['shot_count']
        self.state.update_shot_count(new_shot_count)
        new_shot_efficiency = 1 - (action['shot_count'] / 5)
        self.state.update_shot_efficiency(max(0, new_shot_efficiency))
        
        # Update session duration
        self.state.secondary_state['session_duration'] += 1
        
        # Update breakthrough potential
        if engine == 2:  # PFC
            new_potential = self.state.secondary_state['breakthrough_potential'] + 0.05
            self.state.secondary_state['breakthrough_potential'] = min(1, new_potential)

    def _calculate_reward(self, action):
        """
        Calculate the reward based on the current state and action taken.

        Args:
            action (dict): A dictionary containing the action parameters.

        Returns:
            float: The calculated reward.
        """
        reward = 0
        
        # Reward for research progress
        progress_reward = self.state.primary_state[self.state.embedding_dim + 1] * 10
        reward += progress_reward
        
        # Reward for user engagement
        engagement_reward = self.state.primary_state[-2] * 5
        reward += engagement_reward
        
        # Penalty for excessive shot count
        shot_penalty = action['shot_count'] * 0.5
        reward -= shot_penalty
        
        # Reward for breakthrough potential
        breakthrough_reward = self.state.secondary_state['breakthrough_potential'] * 15
        reward += breakthrough_reward
        
        # Penalty for very high query complexity
        complexity = self.state.primary_state[-1]
        if complexity > 0.8:
            complexity_penalty = (complexity - 0.8) * 5
            reward -= complexity_penalty
        
        # Bonus for balanced FE-IE usage
        fe_ie_balance = action['fe_ie_balance'] / 9.0
        balance_bonus = (1 - abs(fe_ie_balance - 0.5)) * 3
        reward += balance_bonus
        
        return reward

class RLOrchestrator:
    """
    Reinforcement Learning Orchestrator for training and prediction using the SRSWTI environment.
    """

    def __init__(self):
        self.env = SRSWTIEnv()
        self.model = PPO("MlpPolicy", self.env, verbose=1)

    def train(self, total_timesteps=100000, callback=None, log_interval=1000):
        """
        Train the RL model.

        Args:
            total_timesteps (int): Total number of timesteps to train for.
            callback (callable): Callback function for logging.
            log_interval (int): Interval for logging.
        """
        self.model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=log_interval)

    def predict(self, observation):
        """
        Make a prediction using the trained model.

        Args:
            observation: The current observation of the environment.

        Returns:
            action: The predicted action to take.
        """
        action, _states = self.model.predict(observation, deterministic=True)
        return action

    def save(self, path):
        """
        Save the trained model.

        Args:
            path (str): Path to save the model to.
        """
        self.model.save(path)

    def load(self, path):
        """
        Load a trained model.

        Args:
            path (str): Path to load the model from.
        """
        self.model = PPO.load(path, env=self.env)

# Example usage
if __name__ == "__main__":
    orchestrator = RLOrchestrator()
    orchestrator.train(total_timesteps=10000)  # Train for 10000 steps

    # Test the trained model
    obs = orchestrator.env.reset()
    for _ in range(1000):
        action = orchestrator.predict(obs)
        obs, reward, done, _ = orchestrator.env.step(action)
        if done:
            obs = orchestrator.env.reset()

    # Save the trained model
    orchestrator.save("srswti_model")