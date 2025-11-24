import os
import pickle
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc
from sklearn.utils import shuffle
from tqdm import tqdm

# --- 1. Data Generation and Preparation ---


class EigenstateDataset:
    """
    Handles the generation, caching, and preparation of the XXZ Hamiltonian eigenstate dataset.
    """

    def __init__(self, L, delta_range, filename="eigenstate_dataset.pkl"):
        self.L = L
        self.delta_range = delta_range
        self.filename = filename
        self.full_dataset = self._load_or_generate_dataset()

    def _create_xxz_hamiltonian(self, delta):
        """Creates the H = XX + YY + Delta * ZZ Hamiltonian for a 1D chain."""
        g = tc.templates.graphs.Line1D(self.L, pbc=True)
        return tc.quantum.heisenberg_hamiltonian(g, hxx=1, hyy=1, hzz=delta)

    def _load_or_generate_dataset(self):
        """Loads the dataset from a pickle file or generates it if not found."""
        if os.path.exists(self.filename):
            print(f"Dataset '{self.filename}' already exists. Loading from file.")
            with open(self.filename, "rb") as f:
                return pickle.load(f)

        print(f"Generating dataset for L={self.L}. This may take some time...")
        # Use TensorFlow backend for this one-time, potentially heavy operation
        K_data = tc.set_backend("tensorflow")
        dataset = {}
        for delta in tqdm(self.delta_range, desc="Generating Eigenstates"):
            h = self._create_xxz_hamiltonian(delta)
            matrix_h = K_data.to_dense(h)
            eigenvalues, eigenstates = K_data.eigh(matrix_h)
            dataset[delta] = {
                "eigenvalues": K_data.numpy(eigenvalues),
                "eigenstates": K_data.numpy(
                    eigenstates
                ).T,  # eigh returns column vectors
            }

        with open(self.filename, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to '{self.filename}'")
        return dataset

    def get_task_data(self, i, j, model_type, val_split=0.2):
        """
        Extracts and prepares training/validation data for a specific task (i, j).

        Args:
            i (int): Index of the first eigenstate.
            j (int): Index of the second eigenstate.
            model_type (str): 'classical' or 'quantum'. Determines data format.
            val_split (float): Fraction of data to use for validation.
        """
        states_i = [
            self.full_dataset[delta]["eigenstates"][i] for delta in self.full_dataset
        ]
        states_j = [
            self.full_dataset[delta]["eigenstates"][j] for delta in self.full_dataset
        ]

        # Labels: state i -> 0, state j -> 1
        x = np.array(states_i + states_j, dtype=np.complex64)
        y = np.array([0] * len(states_i) + [1] * len(states_j), dtype=np.int32)

        # Shuffle data consistently
        x_shuffled, y_shuffled = shuffle(x, y, random_state=42)

        # Split into training and validation sets
        split_idx = int(len(x_shuffled) * (1 - val_split))
        x_train_raw, x_val_raw = x_shuffled[:split_idx], x_shuffled[split_idx:]
        y_train, y_val = y_shuffled[:split_idx], y_shuffled[split_idx:]

        if model_type == "classical":
            # Classical model needs real-valued input: [real_part, imag_part]
            x_train = np.concatenate(
                [np.real(x_train_raw), np.imag(x_train_raw)], axis=1
            )
            x_val = np.concatenate([np.real(x_val_raw), np.imag(x_val_raw)], axis=1)
            return x_train, y_train, x_val, y_val
        else:  # quantum model takes complex states directly
            return x_train_raw, y_train, x_val_raw, y_val


# --- 2. Model Definitions ---


class ClassicalClassifier:
    """A simple Keras-based MLP classifier."""

    def __init__(self, input_dim):
        self.model = self._create_model(input_dim)

    def _create_model(self, input_dim):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model


class QuantumClassifier:
    """A JAX-based Parameterized Quantum Circuit (PQC) classifier."""

    def __init__(self, L, circuit_depth, random_seed=42):
        self.L = L
        self.circuit_depth = circuit_depth
        self.key = jax.random.PRNGKey(random_seed)
        self.params = self._initialize_params()
        self.K = tc.set_backend("jax")
        self._setup_training_routines()

    def _initialize_params(self):
        self.key, subkey = jax.random.split(self.key)
        shape = (self.circuit_depth, self.L - 1, 15)  # 15 params for each SU(4) gate
        return jax.random.uniform(subkey, shape=shape, minval=-0.1, maxval=0.1)

    def _build_ansatz(self, params, inputs):
        c = tc.Circuit(self.L, inputs=inputs)
        for d in range(self.circuit_depth):
            for i in range(self.L - 1):
                c.su4(i, i + 1, theta=params[d, i])
        return c

    def _predict_logit(self, params, x_input):
        c = self._build_ansatz(params, x_input)
        return self.K.real(c.expectation_ps(z=[0]))  # Pauli Z on the first qubit

    def _setup_training_routines(self):
        def loss_fn(p, x, y):
            logits = self._predict_logit(p, x)
            return optax.sigmoid_binary_cross_entropy(logits, y)

        def acc_fn(p, x, y):
            logits = self._predict_logit(p, x)
            predictions = (logits > 0).astype(jnp.int32)
            return jnp.mean(predictions == y)

        self.value_and_grad_fn = self.K.jit(
            self.K.vectorized_value_and_grad(loss_fn, vectorized_argnums=(1, 2))
        )
        self.compute_accuracy = self.K.jit(
            self.K.vmap(acc_fn, vectorized_argnums=(1, 2))
        )


# --- 3. Unified Experiment Runner ---


def run_experiment(model_type, tasks, dataset_handler, L, epochs, batch_size):
    """
    Runs a continual learning experiment for a given model type.

    Args:
        model_type (str): 'classical' or 'quantum'.
        tasks (list): A list of tuples, where each tuple is a classification task (i, j).
        dataset_handler (EigenstateDataset): The data handler object.
        L (int): System size (number of qubits).
        epochs (int): Number of training epochs per task.
        batch_size (int): Batch size for training.

    Returns:
        list: A list of final validation accuracies for each task.
    """
    print(
        f"\n--- Running Experiment: Continual Learning with {model_type.capitalize()} Model ---"
    )

    validation_accuracies = []

    # --- Model and Optimizer Initialization ---
    if model_type == "classical":
        tc.set_backend("tensorflow")
        num_states = 2**L
        input_dim = 2 * num_states  # Real part + Imaginary part
        model = ClassicalClassifier(input_dim).model
    else:  # quantum
        tc.set_backend("jax")
        # Quantum circuit depth is fixed at 6 as in the original script
        model = QuantumClassifier(L=L, circuit_depth=6)
        optimizer = optax.adam(learning_rate=2e-3)
        params = model.params
        opt_state = optimizer.init(params)

    # --- Continual Learning Loop ---
    for task in tqdm(tasks, desc=f"Training {model_type.capitalize()} Model"):
        i, j = task
        x_train, y_train, x_val, y_val = dataset_handler.get_task_data(i, j, model_type)

        if model_type == "classical":
            history = model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_val, y_val),
                verbose=0,
            )
            final_val_acc = history.history["val_accuracy"][-1]
        else:  # quantum
            for _ in range(epochs):
                # Use tf.data for efficient batching and shuffling
                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
                train_dataset = train_dataset.shuffle(buffer_size=len(x_train)).batch(
                    batch_size
                )

                for x_batch, y_batch in train_dataset:
                    x_batch_np, y_batch_np = x_batch.numpy(), y_batch.numpy()
                    _, grads = model.value_and_grad_fn(params, x_batch_np, y_batch_np)
                    updates, opt_state = optimizer.update(grads, opt_state)
                    params = optax.apply_updates(params, updates)

            final_val_acc = float(
                model.K.mean(model.compute_accuracy(params, x_val, y_val))
            )

        validation_accuracies.append(final_val_acc)

    return validation_accuracies


# --- 4. Main Execution Block ---

if __name__ == "__main__":
    L = 10  # System size (number of qubits)
    NUM_TASKS = 2000  # Number of sequential tasks to learn
    EPOCHS_PER_TASK = 5  # Training epochs for each new task
    BATCH_SIZE = 64  # Batch size for training

    print("--- Configuration ---")
    print(f"System Size (L): {L}")
    print(f"Number of Tasks: {NUM_TASKS}")
    print(f"Epochs per Task: {EPOCHS_PER_TASK}")
    print(f"Batch Size: {BATCH_SIZE}")

    # --- Data and Task Generation ---
    DELTA_RANGE = np.arange(-2.0, 2.02, 0.02)
    dataset_filename = "eigenstate_dataset.pkl"
    dataset_handler = EigenstateDataset(L, DELTA_RANGE, filename=dataset_filename)

    num_states = 2**L
    np.random.seed(42)  # for reproducible task sequence
    tasks = [
        tuple(np.random.choice(range(num_states), 2, replace=False))
        for _ in range(NUM_TASKS)
    ]
    print(f"\nGenerated task sequence of {len(tasks)} tasks.")

    # --- Run Both Experiments ---
    classical_accuracies = run_experiment(
        "classical", tasks, dataset_handler, L, EPOCHS_PER_TASK, BATCH_SIZE
    )
    quantum_accuracies = run_experiment(
        "quantum", tasks, dataset_handler, L, EPOCHS_PER_TASK, BATCH_SIZE
    )

    # --- Save Results ---
    np.save("classical_validation_accuracies.npy", np.array(classical_accuracies))
    np.save("quantum_validation_accuracies.npy", np.array(quantum_accuracies))

    print("\n--- Experiments Finished ---")
    print("Classical model accuracies saved to 'classical_validation_accuracies.npy'")
    print("Quantum model accuracies saved to 'quantum_validation_accuracies.npy'")
