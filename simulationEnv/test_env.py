# simulationEnv/test_env.py

import numpy as np
from rl_orchestrator import SRSWTIEnv

def print_state(obs):
    print("\nCurrent State:")
    print(f"Query Embedding (first 5 dims): {obs[:5]}")
    print(f"User Expertise: {obs[384]:.2f}")
    print(f"Research Progress: {obs[385]:.2f}")
    print(f"Novelty Score: {obs[386]:.2f}")
    print(f"Shot Count: {obs[-8]:.0f}")
    print(f"Shot Efficiency: {obs[-6]:.2f}")
    print(f"User Engagement: {obs[-5]:.2f}")
    print(f"Query Complexity: {obs[-4]:.2f}")
    print(f"Session Duration: {obs[-2]:.0f}")
    print(f"Breakthrough Potential: {obs[-1]:.2f}")

def print_action(action):
    engines = ['FE', 'IE', 'PFC']
    print("\nAction taken:")
    print(f"Engine: {engines[action[0]]}")
    print(f"FE-IE Balance: {action[1]/9:.2f}")
    print(f"Worker: {action[2] if action[0] == 0 else action[3] if action[0] == 1 else action[4]}")
    print(f"Query Refinement: {'None' if action[5] == 0 else 'Expand' if action[5] == 1 else 'Narrow'}")
    print(f"Shot Count: {action[6]}")

def test_environment(episodes=3, steps_per_episode=20):
    env = SRSWTIEnv()

    for episode in range(episodes):
        print(f"\n{'='*50}\nEpisode {episode+1}\n{'='*50}")
        obs = env.reset()
        total_reward = 0

        for step in range(steps_per_episode):
            print(f"\n{'-'*40}\nStep {step+1}")
            
            print_state(obs)
            
            action = env.action_space.sample()  # Random action
            print_action(action)
            
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            
            print(f"\nReward: {reward:.2f}")
            print(f"Total Reward: {total_reward:.2f}")
            
            if done:
                print("\nEpisode finished early.")
                break

        print(f"\nEpisode {episode+1} finished. Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    test_environment()