import time
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc
from tqdm import tqdm

# --- 1. Global Hyperparameters ---

# Shared Hyperparameters
NUM_TASKS = 1000
BATCH_SIZE = 128
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Classical Model Hyperparameters
CLASSICAL_LR = 0.001
CLASSICAL_INPUT_DIM = 784
MLP_NW = 16
EPOCHS_PER_TASK = 1

# Quantum Model Hyperparameters
QUANTUM_LR = 0.03
QUANTUM_EPOCHS_PER_TASK = 5  # Quantum model trains for more epochs per task
L_QUBITS = 10
N_FEATURES = 1024  # Padded dimension for quantum input
QNN_DEPTH = 4

# --- 2. Data Preparation ---


def prepare_mnist_data(model_type):
    """Loads and preprocesses MNIST data according to the model's requirements."""
    print(f"\n1. Loading and preprocessing MNIST data for '{model_type}' model...")
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = (
        tf.keras.datasets.mnist.load_data()
    )

    x_train_flat = x_train_raw.reshape(-1, 784).astype("float32") / 255.0
    x_test_flat = x_test_raw.reshape(-1, 784).astype("float32") / 255.0

    if model_type == "classical":
        num_features = CLASSICAL_INPUT_DIM
        x_train, y_train = x_train_flat, y_train_raw
        x_test, y_test = x_test_flat, y_test_raw
    else:  # quantum
        num_features = N_FEATURES
        print(
            f"   Padding data from {CLASSICAL_INPUT_DIM} to {N_FEATURES} dimensions and L2-normalizing..."
        )

        def pad_and_normalize(data):
            padding = np.zeros((data.shape[0], N_FEATURES - 784), dtype=np.float32)
            padded_data = np.concatenate([data, padding], axis=1)
            norm = np.linalg.norm(padded_data, axis=1, keepdims=True)
            return padded_data / (norm + 1e-9)

        x_train, y_train = pad_and_normalize(x_train_flat), y_train_raw
        x_test, y_test = pad_and_normalize(x_test_flat), y_test_raw

    print(f"2. Generating permutation maps for {NUM_TASKS} tasks...")
    permutations = [np.random.permutation(num_features) for _ in range(NUM_TASKS)]

    print("   Data preparation complete.")
    return (x_train, y_train), (x_test, y_test), permutations


# --- 3. Classical Model & Analysis Functions ---


def create_classical_model(input_shape, learning_rate):
    """Creates a simple Keras MLP model."""
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(input_shape,)),
            tf.keras.layers.Dense(MLP_NW, activation="relu", name="hidden_layer"),
            tf.keras.layers.Dense(10, activation="softmax", name="output_layer"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def analyze_classical_model(model, x_data, y_data):
    """Calculates average weight and gradient norms for the classical model."""
    # Weight norm calculation
    all_weights = model.get_weights()
    weight_matrix_norms = [np.linalg.norm(w) for w in all_weights if w.ndim > 1]
    avg_weight_norm = np.mean(weight_matrix_norms) if weight_matrix_norms else 0.0

    # Gradient norm calculation
    sample_size = min(BATCH_SIZE, x_data.shape[0])
    indices = np.random.choice(x_data.shape[0], sample_size, replace=False)
    x_batch, y_batch = x_data[indices], y_data[indices]

    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=False)
        loss = model.compiled_loss(y_batch, predictions)

    grads = tape.gradient(loss, model.trainable_variables)

    grad_tensor_norms = [tf.linalg.norm(g).numpy() for g in grads if g is not None]
    avg_grad_norm = np.mean(grad_tensor_norms) if grad_tensor_norms else 0.0

    return avg_weight_norm, avg_grad_norm


# --- 4. Quantum Model & Analysis Functions ---


class SU4QuantumClassifier:
    """JAX-based Quantum Classifier for MNIST."""

    def __init__(self, L, d_main, n_classes=10):
        self.K = tc.set_backend("jax")
        tc.set_dtype("complex64")
        self.L = L
        self.d_main = d_main
        self.n_classes = n_classes
        key = jax.random.PRNGKey(RANDOM_SEED)
        key_main_even, key_main_odd = jax.random.split(key, 2)
        self.params = {
            "main_even": jax.random.uniform(
                key_main_even, shape=[d_main, L // 2, 15], minval=0, maxval=2 * jnp.pi
            ),
            "main_odd": jax.random.uniform(
                key_main_odd,
                shape=[d_main, (L - 1) // 2, 15],
                minval=0,
                maxval=2 * jnp.pi,
            ),
        }

    def _build_ansatz(self, params, inputs):
        c = tc.Circuit(self.L, inputs=inputs)
        for d in range(self.d_main):
            for i in range(0, self.L, 2):
                c.su4(i, (i + 1) % self.L, theta=params["main_even"][d, i // 2])
            for i in range(1, self.L - 1, 2):
                c.su4(i, (i + 1) % self.L, theta=params["main_odd"][d, i // 2])
        return c

    def _predict_logits(self, params, x_input):
        final_circuit = self._build_ansatz(params, x_input)
        probabilities = final_circuit.probability()
        epsilon = 1e-7
        probs_clipped = jnp.clip(probabilities, epsilon, 1 - epsilon)
        logits = jnp.log(probs_clipped / (1 - probs_clipped))
        logits = jnp.real(logits[: self.n_classes])
        return logits + 1e-9
        # c = self._build_ansatz(params, x_input)
        # return self.K.stack(
        #     [self.K.real(c.expectation_ps(z=[i])) for i in range(self.n_classes)]
        # )

    def setup_routines(self):
        self.batch_predict_logits = self.K.vmap(
            self._predict_logits, vectorized_argnums=1
        )

        def loss_fn(p, x, y_onehot):
            logits = self.batch_predict_logits(p, x)
            return jnp.mean(optax.softmax_cross_entropy(logits, y_onehot))

        def acc_fn(p, x, y_labels):
            logits = self.batch_predict_logits(p, x)
            predictions = jnp.argmax(logits, axis=1)
            return jnp.mean(predictions == y_labels)

        self.value_and_grad_fn = self.K.jit(self.K.value_and_grad(loss_fn))
        self.compute_accuracy = self.K.jit(acc_fn)


def analyze_quantum_model(model, params, x_data, y_data_onehot):
    """Calculates average parameter and gradient norms for the quantum model."""
    _, grads = model.value_and_grad_fn(params, x_data, y_data_onehot)

    def calculate_average_norm(pytree):
        leaf_norms = jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), pytree)
        )
        return jnp.mean(jnp.array(leaf_norms)) if leaf_norms else 0.0

    avg_param_norm = float(calculate_average_norm(params))
    avg_grad_norm = float(calculate_average_norm(grads))
    return avg_param_norm, avg_grad_norm


# --- 5. Unified Experiment Runner ---


def run_experiment(model_type):
    """Runs the full continual learning experiment for a given model type."""
    if model_type not in ["classical", "quantum"]:
        raise ValueError("model_type must be 'classical' or 'quantum'")

    print(f"\n{'='*20} Starting Experiment: {model_type.capitalize()} Model {'='*20}")

    # --- Data and Permutations ---
    (x_train, y_train), (x_test, y_test), permutations = prepare_mnist_data(model_type)

    results = {"accuracies": [], "weight_norms": [], "grad_norms": []}

    # --- Model Initialization ---
    if model_type == "classical":
        model = create_classical_model(CLASSICAL_INPUT_DIM, CLASSICAL_LR)
        model.summary()
        epochs = EPOCHS_PER_TASK
    else:  # quantum
        model = SU4QuantumClassifier(L=L_QUBITS, d_main=QNN_DEPTH)
        model.setup_routines()
        epochs = QUANTUM_EPOCHS_PER_TASK

        y_train_onehot = tf.one_hot(y_train, 10).numpy()

    # --- Continual Learning Loop ---
    start_time_exp = time.time()
    for task_id in tqdm(
        range(NUM_TASKS), desc=f"Training {model_type.capitalize()} Model"
    ):

        perm = permutations[task_id]
        x_train_permuted = x_train[:, perm]
        x_test_permuted = x_test[:, perm]

        # Analyze model state before training
        if model_type == "classical":
            weight_norm, grad_norm = analyze_classical_model(
                model, x_train_permuted, y_train
            )
        else:  # quantum
            sample_x = x_train_permuted[:BATCH_SIZE]
            sample_y = y_train_onehot[:BATCH_SIZE]
            weight_norm, grad_norm = analyze_quantum_model(
                model, model.params, sample_x, sample_y
            )

        results["weight_norms"].append(weight_norm)
        results["grad_norms"].append(grad_norm)

        # Train the model
        if model_type == "classical":
            model.fit(
                x_train_permuted,
                y_train,
                batch_size=BATCH_SIZE,
                epochs=epochs,
                verbose=0,
            )
        else:  # quantum
            dataset = tf.data.Dataset.from_tensor_slices(
                (x_train_permuted, y_train_onehot)
            )
            dataset = dataset.shuffle(buffer_size=len(x_train), seed=RANDOM_SEED).batch(
                BATCH_SIZE
            )
            optimizer = optax.adam(learning_rate=QUANTUM_LR)
            opt_state = optimizer.init(model.params)
            for _ in range(epochs):
                for x_batch, y_batch in dataset:
                    _, grads = model.value_and_grad_fn(
                        model.params, x_batch.numpy(), y_batch.numpy()
                    )
                    updates, opt_state = optimizer.update(grads, opt_state)
                    model.params = optax.apply_updates(model.params, updates)

        # Evaluate the model
        if model_type == "classical":
            _, accuracy = model.evaluate(x_test_permuted, y_test, verbose=0)
        else:  # quantum
            accuracy = float(
                model.compute_accuracy(model.params, x_test_permuted, y_test)
            )

        results["accuracies"].append(accuracy)

    end_time_exp = time.time()
    print(
        f"{'='*20} Experiment Finished: Time taken {end_time_exp - start_time_exp:.2f} seconds {'='*20}"
    )

    # Convert lists to numpy arrays for saving
    for key in results:
        results[key] = np.array(results[key])

    return results


# --- 6. Main Execution Block ---
if __name__ == "__main__":
    total_start_time = time.time()

    # Run classical experiment
    classical_results = run_experiment("classical")
    np.savez("classical_model_results.npz", **classical_results)
    print("Classical model results saved to 'classical_model_results.npz'")

    # Run quantum experiment
    quantum_results = run_experiment("quantum")
    np.savez("quantum_model_results.npz", **quantum_results)
    print("Quantum model results saved to 'quantum_model_results.npz'")

    total_end_time = time.time()
    print(
        f"\nAll experiments completed. Total time: {(total_end_time - total_start_time) / 60:.2f} minutes"
    )
