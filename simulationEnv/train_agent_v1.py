import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from srswti_env import SRSWTIEnv

def train_and_evaluate(total_timesteps, eval_episodes, seed):
    # Create a single environment
    env = SRSWTIEnv()

    # Create the PPO agent
    model = PPO("MlpPolicy", env, verbose=1, seed=seed)

    # Train the agent
    model.learn(total_timesteps=total_timesteps)

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=eval_episodes)

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return model, env

def test_agent(model, env, num_episodes):
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0

        print(f"\nEpisode {episode + 1}")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward.sum() if isinstance(reward, np.ndarray) else reward
            step += 1

            action_str = action.tolist() if isinstance(action, np.ndarray) else action
            reward_str = reward.sum() if isinstance(reward, np.ndarray) else reward
            print(f"Step {step}: Action={action_str}, Reward={reward_str:.2f}")

        print(f"Episode finished. Total reward: {total_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate PPO agent on SRSWTI Environment")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps for training")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of episodes for evaluation")
    parser.add_argument("--test_episodes", type=int, default=3, help="Number of episodes for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    model, env = train_and_evaluate(args.timesteps, args.eval_episodes, args.seed)
    test_agent(model, env, args.test_episodes)