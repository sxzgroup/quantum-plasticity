# Quantum Neural Networks Preserve Plasticity in Continual Learning

[![Framework: TensorCircuit](https://img.shields.io/badge/Framework-TensorCircuit_NG-blue.svg)](https://github.com/tensorcircuit/tensorcircuit-ng)
[![arXiv](https://img.shields.io/badge/arXiv-2511.17228-blue.svg)](https://arxiv.org/abs/2511.17228)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official implementation for the paper [**Intrinsic preservation of plasticity in continual quantum learning**](https://arxiv.org/abs/2511.17228).

In this work, we demonstrate that standard deep learning methods suffer from a fundamental "loss of plasticity" in continual learning settings, wherein networks gradually lose their ability to learn from new data. In contrast, **Deep Quantum Neural Networks (QNNs)** naturally overcome this limitation due to the intrinsic geometric constraints of their unitary parameter manifold. We validate this advantage systematically across four diverse experimental settings ranging from classical computer vision to deep reinforcement learning and quantum-native tasks.

## 📂 Project Structure

The codebase is organized into four independent scripts in `/src`, each corresponding to a major experiment in the paper:

- **`mnist.py`**: Supervised continual learning on **Permuted MNIST**.
- **`cifar.py`**: Supervised continual learning on **Split CIFAR-100**.
- **`rl.py`**: Deep Reinforcement Learning on **MuJoCo Ant-v4**.
- **`qdata.py`**: Continual learning on **Quantum-Native Data**.

## 🛠️ Installation & Requirements

This project relies on [**TensorCircuit-NG**](https://github.com/tensorcircuit/tensorcircuit-ng) for quantum simulation, **JAX** and **TensorFlow** for automatic differentiation/training, and **Stable-Baselines3** for reinforcement learning. Specifically, the scale of experiments in this work is only possible with the help of high performance TensorCircuit-NG.

### Prerequisites

- Python 3.10+
- CUDA-enabled GPU (Recommended for Deep QNN simulation)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/sxzgroup/quantum-plasticity.git
   cd quantum-plasticity
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # the below is only needed for the RL experiment
   pip install "stable-baselines3[extra]" "gymnasium[mujoco]"
   ```

## 🚀 Usage & Experiments

Each script is self-contained including data preparation and training for both classical and quantum models. Below are the details for reproducing the results reported in the paper. The results will be automatically saved to `.npz` files.

### 1. Permuted MNIST (Supervised)

Compares a Classical MLP against a QNN on 1,000 sequential tasks where pixels are randomly permuted.

- **Model:** MLP vs. Deep QNN (Hardware-Efficient Ansatz).
- **Metrics:** Test Accuracy, Weight Norms, Gradient Norms.

```bash
python mnist.py
```

_Output:_ Saves `classical_model_results.npz` and `quantum_model_results.npz` containing accuracy and norm trajectories.

### 2. Split CIFAR-100

Evaluates scalability on 3,000 binary classification tasks derived from CIFAR-100. Includes **Fisher Information Matrix (FIM)** analysis.

- **Model:** Deep QNN (up to 30 layers) vs. Wide MLP.
- **Key Metric:** Trace of the Fisher Information Matrix calculated on a fixed probe dataset to measure effective learnability.

```bash
python cifar.py
```

_Output:_ Saves results including FIM traces to `classical_cifar100_results.npz` and `quantum_cifar100_results.npz`.

### 3. Deep Reinforcement Learning (Ant-v4)

A challenging continuous control task where the agent must adapt to a composite reward function (velocity, survival, control cost).

- **Agent:** Proximal Policy Optimization (PPO).
- **Comparison:** Standard MLP Policy vs. Hybrid Quantum-PPO Policy.
- **Environment:** `Ant-v4` (MuJoCo).

You nedd install `pip install "stable-baselines3[extra]" "gymnasium[mujoco]"` for this RL task, better in a separate environment.

```bash
python rl.py
```

_Output:_ Saves training rewards in `.npy` files and logs.

### 4. Quantum-Native Data

Binary classification of many-body eigenstates generated from a 1D Heisenberg XXZ Hamiltonian with varying anisotropy $\Delta$.

- **Data:** 2,000 tasks generated from quantum phase transitions.
- **Finding:** Demonstrates "Dual Advantage" — QNNs learn better (higher accuracy) and longer (no plasticity loss) than classical models on quantum data.

```bash
python qdata.py
```

_Output:_ Generates dataset cache `eigenstate_dataset.pkl` (first run) and saves validation accuracies to `.npy` files.

---

## 📜 Citation

If you find this code useful in your research, please consider citing our paper:

```bibtex
@article{plasticity2025,
  title={Intrinsic preservation of plasticity in continual quantum learning},
  author={Shi-Xin Zhang and Yu-Qin Chen},
  journal={arXiv preprint arXiv:2511.17228},
  year={2025}
}
```

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file.
