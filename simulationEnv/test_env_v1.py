# simulationEnv/test_env.py

import argparse
import numpy as np
import sys
import os
from srswti_env import SRSWTIEnv

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()

def describe_state(state):
    """Provide a detailed description of the current state."""
    query_embedding = state[:5]
    user_expertise = state[5]
    active_engine = ['None', 'FE', 'IE', 'PFC'][int(state[6])]
    task_progress = state[7]
    kb_relevance = state[8]
    
    return f"""
    Query Embedding: {query_embedding} (Represents the semantic meaning of the current query)
    User Expertise: {user_expertise:.2f} (0: Novice, 1: Expert)
    Active Engine: {active_engine}
    Task Progress: {task_progress:.2f} (0: Not started, 1: Completed)
    Knowledge Base Relevance: {kb_relevance:.2f} (0: Not relevant, 1: Highly relevant)
    """

def describe_action(action):
    """Provide a description of the chosen action."""
    actions = ['Query FE', 'Activate IE', 'Engage PFC', 'Refine Query']
    return f"Action: {actions[action]} ({action})"

def get_next_episode_number():
    i = 1
    while os.path.exists(f"episodeData_{i}.txt"):
        i += 1
    return i

def main(episodes, max_steps, seed):
    np.random.seed(seed)
    env = SRSWTIEnv()
    env.max_steps = max_steps

    episode_rewards = []
    episode_steps = []
    successful_episodes = 0

    # Set up Tee to write to both stdout and file
    episode_number = get_next_episode_number()
    tee = Tee(sys.stdout, open(f"episodeData_{episode_number}.txt", "w"))
    sys.stdout = tee

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        print(f"\n{'='*50}\nEpisode: {episode + 1}\n{'='*50}")
        print("Initial State:")
        print(describe_state(state))
        
        while not done:
            print(f"\n{'-'*30}\nStep {step + 1}:")
            
            # Choose a random action
            action = env.action_space.sample()
            print(describe_action(action))
            
            # Take the action
            next_state, reward, done, _ = env.step(action)
            
            print("\nResult:")
            print(describe_state(next_state))
            print(f"Reward: {reward:.2f}")
            
            if done:
                if env.current_step >= env.max_steps:
                    print("\nEpisode terminated: Maximum steps reached")
                elif next_state[7] >= 1.0:
                    print("\nEpisode terminated: Task completed successfully")
            
            total_reward += reward
            state = next_state
            step += 1
        
        print(f"\nEpisode {episode + 1} finished.")
        print(f"Total steps: {step}")
        print(f"Total reward: {total_reward:.2f}")

        episode_rewards.append(total_reward)
        episode_steps.append(step)
        if state[7] >= 1.0:
            successful_episodes += 1

    print("\nOverall Statistics:")
    print(f"Average Reward per Episode: {np.mean(episode_rewards):.2f}")
    print(f"Average Steps per Episode: {np.mean(episode_steps):.2f}")
    print(f"Successful Episodes: {successful_episodes}/{episodes} ({successful_episodes/episodes*100:.2f}%)")

    # Restore stdout
    sys.stdout = sys.__stdout__
    print(f"Output has been saved to episodeData_{episode_number}.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SRSWTI Environment")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    main(args.episodes, args.max_steps, args.seed)