import os
import time
import numpy as np
import jax
import torch
from torch import nn
import gymnasium as gym
import tensorcircuit as tc
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# --- 1. Global Setup and Quantum Model Definition ---

# 1.1 Set backend and device for reproducibility and performance
os.environ["SB3_LOGGING_FORMAT"] = "stdout"
K = tc.set_backend("jax")
PT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using PyTorch device: {PT_DEVICE}")
print(f"JAX backend default device: {jax.default_backend()}")


# 1.2 JAX JIT quantum function for feature extraction
@K.jit
def su4_only_extractor(x, weights):
    """
    Quantum circuit for feature extraction using SU(4) gates.
    """
    n_qubits = 9

    # n_layers is inferred from the weights shape
    def scan_body(carry_state, params_su4_layer):
        c_layer = tc.Circuit(n_qubits, inputs=carry_state)
        for i in range(n_qubits):
            c_layer.su4(i, (i + 1) % n_qubits, theta=params_su4_layer[i])
        return c_layer.state(), None

    c = tc.Circuit(n_qubits, inputs=x)
    initial_state = c.state()
    final_state, _ = jax.lax.scan(scan_body, initial_state, weights)
    final_circuit = tc.Circuit(n_qubits, inputs=final_state)
    return K.tanh(5.0 * K.real(final_circuit.probability()))


# 1.3 PyTorch feature extractor module wrapping the quantum circuit
class SU4QuantumExtractorModule(nn.Module):
    """
    A PyTorch nn.Module that acts as a feature extractor by wrapping the JAX-based quantum circuit.
    """

    def __init__(self, observation_space: gym.spaces.Space):
        super().__init__()
        self.n_qubits = 9
        self.n_layers = 12
        self.padded_dim = 2**self.n_qubits
        self.quantum_layer = tc.TorchLayer(
            su4_only_extractor,
            weights_shape=[
                self.n_layers,
                self.n_qubits,
                15,
            ],  # Match the weight shape with the JAX function
            use_vmap=True,
            use_jit=True,
            enable_dlpack=(PT_DEVICE.type != "cpu"),
        )
        self.features_dim = 2**self.n_qubits

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        # Pad features to match the dimension of the quantum state vector
        padded_features = torch.zeros(
            batch_size, self.padded_dim, device=features.device
        )
        padded_features[:, : features.shape[1]] = features
        # Normalize the input vector to be a valid quantum state amplitude vector
        norm = torch.linalg.norm(padded_features, ord=2, dim=1, keepdim=True) + 1e-8
        normalized_features = padded_features / norm
        return self.quantum_layer(normalized_features)


# --- 2. Reusable Experiment Components ---

# 2.1 Hyperparameters
TOTAL_STEPS = 30_000_000
N_ENVS = 16
EVAL_FREQ = 200_000
N_EVAL_EPISODES = 10
N_STEPS_PER_ENV = int(TOTAL_STEPS / N_ENVS)
BATCH_SIZE = 128
N_EPOCHS = 10
GAE_LAMBDA = 0.95
GAMMA = 0.99
CLIP_RANGE = 0.2
ENV_ID = "Ant-v4"
np.random.seed(42)


# 2.2 Callback for logging training rewards
class RewardLoggerCallback(BaseCallback):
    """
    A custom callback to log rewards during training for each episode.
    """

    def __init__(self):
        super().__init__()
        self.rewards = []
        self.timesteps = []

    def _on_step(self) -> bool:
        # Check for completed episodes in all parallel environments
        for i in range(self.training_env.num_envs):
            if self.locals["dones"][i]:
                info = self.locals["infos"][i]
                if "episode" in info:
                    self.rewards.append(info["episode"]["r"])
                    self.timesteps.append(self.num_timesteps)
        return True

    def get_results(self):
        return np.array(self.timesteps), np.array(self.rewards)


# --- 3. Unified Experiment Runner ---


def run_experiment(experiment_type: str):
    """
    Runs a PPO experiment, configurable for 'classical' or 'quantum' models.

    Args:
        experiment_type (str): Type of experiment, either 'classical' or 'quantum'.
    """
    if experiment_type not in ["classical", "quantum"]:
        raise ValueError("experiment_type must be 'classical' or 'quantum'")

    print(
        f"\n{'='*20} Starting {experiment_type.capitalize()} PPO Experiment ({N_ENVS} parallel envs) {'='*20}"
    )

    # --- Setup logging and directories ---
    log_dir = f"./ppo_ant_{experiment_type}_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # --- Environment Setup ---
    train_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID)) for _ in range(N_ENVS)])
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID)) for _ in range(N_ENVS)])

    # --- Callback Configuration ---
    reward_callback = RewardLoggerCallback()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )
    callback_list = CallbackList([reward_callback, eval_callback])

    # --- Model Configuration based on experiment type ---
    if experiment_type == "quantum":
        policy_kwargs = dict(
            features_extractor_class=SU4QuantumExtractorModule,
            net_arch=dict(pi=[], vf=[]),  # No extra layers after the quantum extractor
        )
        learning_rate = 0.002
    else:  # classical
        policy_kwargs = dict(net_arch=dict(pi=[1024, 256], vf=[256, 256]))
        learning_rate = 0.0001

    # --- PPO Model Initialization ---
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=N_STEPS_PER_ENV,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=PT_DEVICE,
        tensorboard_log=os.path.join(log_dir, "tb_logs"),
    )

    # --- Training ---
    print(f"Starting training for {TOTAL_STEPS} timesteps...")
    start_time = time.time()
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=callback_list,
        tb_log_name=f"ppo_ant_{experiment_type}",
        progress_bar=False,
    )
    end_time = time.time()
    print(f"Training finished in {(end_time - start_time) / 3600:.2f} hours.")

    # --- Collect and Save Results ---
    print("Collecting and saving results...")
    train_timesteps, train_rewards = reward_callback.get_results()
    if len(train_timesteps) > 0:
        training_results = np.array([train_timesteps, train_rewards])
        train_save_path = f"{experiment_type}_training_results.npy"
        np.save(train_save_path, training_results)
        print(f"Training results saved to {train_save_path}")

    # 2. Process and save evaluation results
    eval_results_path = os.path.join(log_dir, "evaluations.npz")
    try:
        eval_data = np.load(eval_results_path)
        eval_timesteps = eval_data["timesteps"]
        mean_eval_rewards = np.mean(eval_data["results"], axis=1)

        evaluation_results = np.array([eval_timesteps, mean_eval_rewards])
        eval_save_path = f"{experiment_type}_evaluation_results.npy"
        np.save(eval_save_path, evaluation_results)
        print(f"Evaluation results saved to {eval_save_path}")

    except FileNotFoundError:
        print(
            f"Warning: Evaluation results file not found at '{eval_results_path}'. No evaluation data will be saved."
        )

    print(f"Results for {experiment_type} experiment saved successfully.")


# --- 4. Main Execution Block ---
if __name__ == "__main__":
    # Run the classical experiment
    run_experiment(experiment_type="classical")

    # Run the quantum experiment
    run_experiment(experiment_type="quantum")
