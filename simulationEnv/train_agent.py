# simulationEnv/train_agent.py

import numpy as np
from rl_orchestrator import RLOrchestrator
from stable_baselines3.common.callbacks import BaseCallback

class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1

        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True

    def on_training_end(self) -> None:
        print("\nTraining completed!")
        print(f"Average episode reward: {np.mean(self.episode_rewards):.2f}")
        print(f"Average episode length: {np.mean(self.episode_lengths):.2f}")

def train_agent(total_timesteps=100000, log_interval=1000):
    orchestrator = RLOrchestrator()
    callback = TrainingCallback()

    print("Starting training...")
    orchestrator.train(total_timesteps=total_timesteps, callback=callback, log_interval=log_interval)

    print("\nTesting trained agent...")
    test_agent(orchestrator, episodes=5)

def test_agent(orchestrator, episodes=5):
    env = orchestrator.env
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0
        print(f"\n{'='*50}\nTest Episode {episode+1}\n{'='*50}")
        while not done:
            action = orchestrator.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            step += 1
            if step % 5 == 0:  # Print every 5 steps to reduce output
                print(f"Step {step}, Reward: {reward:.2f}")
        print(f"Episode finished. Total steps: {step}, Total reward: {total_reward:.2f}")

if __name__ == "__main__":
    train_agent()