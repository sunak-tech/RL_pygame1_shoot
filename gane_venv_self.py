import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.envs.registration import register
import time # For potential delays or info
import os

# Import the custom environment class
from game_py.MyGameEnv import MyGameEnv # Make sure MyGameEnv.py is in the same directory or PYTHONPATH

# Register the custom environment
ENVIRONMENT_ID = "MyGameEnv-v0"
if ENVIRONMENT_ID not in gym.envs.registry: # Check if already registered
    register(
        id=ENVIRONMENT_ID,
        entry_point="MyGameEnv:MyGameEnv", # Format: module_name:ClassName
    )
else:
    print(f"Environment '{ENVIRONMENT_ID}' is already registered.")


def train_agent(total_timesteps=20000, load_existing_model=False, model_path="ppo_mygame"):
    print("--- Training Phase ---")
    # Create the environment for training (render_mode can be None or "rgb_array" for faster training)
    # Pass any custom args for MyGameEnv here if needed, e.g., initial_enemies
    env = gym.make(ENVIRONMENT_ID, render_mode=None, initial_enemies=3, max_episode_steps=1000)
    env = DummyVecEnv([lambda: env]) # Wrap it for Stable Baselines3

    if load_existing_model and os.path.exists(f"{model_path}.zip"):
        print(f"Loading existing model from {model_path}.zip")
        model = PPO.load(model_path, env=env)
    else:
        print("Creating a new model.")
        # MlpPolicy is suitable for flat observation spaces like ours
        # You can tune hyperparameters here (learning_rate, n_steps, batch_size, etc.)
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_game_tensorboard/")

    #print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, log_interval=1) # Log more frequently
    model.save(model_path)
    #print(f"Training complete. Model saved to {model_path}.zip")
    env.close()


def test_agent(model_path="ppo_mygame", num_episodes=5):
    print("\n--- Testing Phase ---")
    # Create the environment for testing with human rendering
    # Ensure any custom args match those the model might expect implicitly or during training setup
    env = gym.make(ENVIRONMENT_ID, render_mode="human", initial_enemies=3, max_episode_steps=1500)
    # No DummyVecEnv needed for single instance testing like this if we manually loop,
    # but if using model.predict with a non-VecEnv, ensure it's handled correctly or wrap it.
    # For simplicity, let's not wrap for manual testing loop.
    # model.predict expects a VecEnv-like observation or a single observation.

    if not os.path.exists(f"{model_path}.zip"):
        print(f"Model not found at {model_path}.zip. Please train the agent first.")
        env.close()
        return

    model = PPO.load(model_path)
    #print(f"Loaded model from {model_path}.zip for testing.")

    for episode in range(num_episodes):
        obs, info = env.reset()
        #print(f"Episode {episode + 1} Reset complete. Initial Obs (first 6): {obs[:6]}, Info: {info}") # ★追加

        terminated = False
        truncated = False
        total_episode_reward = 0
        step_count = 0 # ★追加: ステップカウンタ
        #print(f"\nStarting Episode {episode + 1}")

        while not (terminated or truncated):
            step_count += 1 # ★追加
            # Agent predicts action for the enemies
            action, _states = model.predict(obs, deterministic=True)

            # ★★★ env.step() の直前直後の情報を徹底的にログ出力 ★★★
            #print(f"  Step {step_count} PRE-STEP : Action to take: {action}, PlayerX_approx: {obs[0]:.2f}")

            # Environment steps based on enemy action (player actions are handled inside env.step)
            obs, reward, terminated, truncated, info = env.step(action)

            # print(f"  Step {step_count} POST-STEP: Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}, Score: {info.get('score', 'N/A')}, Level: {info.get('level', 'N/A')}, PlayerX_approx: {obs[0]:.2f}")
            # if terminated:
            #     print(f"    Terminated reason: (Check MyGameEnv logic for setting terminated=True)")
            # if truncated:
            #     print(f"    Truncated reason: (Likely max_episode_steps reached or user_quit)")


            # env.render() is called within env.step() when render_mode="human"
            # If not, or if you want to control rendering explicitly:
            # env.render() # uncomment if render is not part of step

            total_episode_reward += reward

            if info.get("user_quit", False):
                print("User quit the game during testing.")
                terminated = True # Force end of testing

        print(f"Episode {episode + 1} Finished.")
        print(f"Total Reward: {total_episode_reward}")
        print(f"Final Score: {info.get('score', 0)}, Level: {info.get('level', 1)}")
        if info.get("user_quit", False):
            break # Exit testing loop if user quit

    env.close()
    print("Testing complete.")


if __name__ == "__main__":
    # --- Control Panel ---
    DO_TRAINING = True # Set to True to train, False to only test
    DO_TESTING = True  # Set to True to test after training (or if DO_TRAINING is False)
    LOAD_EXISTING_MODEL_FOR_TRAINING = False # If True and DO_TRAINING is True, continues training
    TRAINING_TIMESTEPS = 30000 # Number of timesteps for training
    MODEL_FILENAME = "ppo_rl_shooter" # Base name for the saved model

    if DO_TRAINING:
        train_agent(total_timesteps=TRAINING_TIMESTEPS,
                    load_existing_model=LOAD_EXISTING_MODEL_FOR_TRAINING,
                    model_path=MODEL_FILENAME)

    if DO_TESTING:
        test_agent(model_path=MODEL_FILENAME, num_episodes=3)

    print("\nProgram finished.")