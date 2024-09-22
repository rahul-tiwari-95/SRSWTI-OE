# simulationEnv/rl_orchestrator.py

import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO

class SRSWTIEnv(gym.Env):
    def __init__(self):
        super(SRSWTIEnv, self).__init__()

        # Define action space
        self.action_space = spaces.MultiDiscrete([
            3,  # Engine selection: 0-FE, 1-IE, 2-PFC
            10, # FE-IE balance: 0-9 (0: full FE, 9: full IE)
            5,  # Worker selection within chosen engine
        ])

        # Define observation space
        self.observation_space = spaces.Dict({
            'query_embedding': spaces.Box(low=-1, high=1, shape=(384,), dtype=np.float32),
            'user_expertise': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'research_progress': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'resource_utilization': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'recent_actions': spaces.MultiDiscrete([3, 10, 5, 3, 10, 5]),  # Last 2 actions
        })

        self.reset()

    def reset(self):
        # Initialize state
        self.state = {
            'query_embedding': np.random.uniform(-1, 1, 384),
            'user_expertise': np.random.uniform(0, 1, 1),
            'research_progress': np.array([0.0]),
            'resource_utilization': np.array([0.0]),
            'recent_actions': np.zeros(6, dtype=int),
        }
        return self.state

    def step(self, action):
        # Unpack action
        engine, fe_ie_balance, worker = action

        # Update state based on action
        self._update_state(engine, fe_ie_balance, worker)

        # Calculate reward
        reward = self._calculate_reward(engine, fe_ie_balance, worker)

        # Check if episode is done
        done = self.state['research_progress'][0] >= 1.0

        return self.state, reward, done, {}

    def _update_state(self, engine, fe_ie_balance, worker):
        # Simulate state changes based on action
        if engine == 0:  # FE
            self.state['research_progress'] += np.random.uniform(0, 0.1)
            self.state['resource_utilization'] += np.random.uniform(0, 0.05)
        elif engine == 1:  # IE
            self.state['research_progress'] += np.random.uniform(0, 0.15)
            self.state['resource_utilization'] += np.random.uniform(0, 0.1)
        else:  # PFC
            self.state['research_progress'] += np.random.uniform(0, 0.05)
            self.state['resource_utilization'] += np.random.uniform(0, 0.02)

        # Update recent actions
        self.state['recent_actions'] = np.roll(self.state['recent_actions'], 3)
        self.state['recent_actions'][:3] = [engine, fe_ie_balance, worker]

        # Ensure values are within bounds
        self.state['research_progress'] = np.clip(self.state['research_progress'], 0, 1)
        self.state['resource_utilization'] = np.clip(self.state['resource_utilization'], 0, 1)

    def _calculate_reward(self, engine, fe_ie_balance, worker):
        # Basic reward function
        reward = self.state['research_progress'][0] * 10 - self.state['resource_utilization'][0] * 5
        
        # Bonus for balanced FE-IE usage
        if 3 <= fe_ie_balance <= 6:
            reward += 2

        return reward

class RLOrchestrator:
    def __init__(self):
        self.env = SRSWTIEnv()
        self.model = PPO("MultiInputPolicy", self.env, verbose=1)

    def train(self, total_timesteps=100000):
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, observation):
        action, _states = self.model.predict(observation, deterministic=True)
        return action

# Usage
orchestrator = RLOrchestrator()
orchestrator.train()

# Example prediction
obs = orchestrator.env.reset()
action = orchestrator.predict(obs)
print(f"Predicted action: {action}")