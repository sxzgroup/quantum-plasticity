import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow informational messages

import time
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc
import gc
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

# --- 1. Global Hyperparameters ---

# Shared Hyperparameters
NUM_TASKS = 30
N_FEATURES = 1024  # 32x32 grayscale image flattened
EPOCHS_PER_TASK = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001
RANDOM_SEED = 42
N_CLASSES_PER_TASK = 2
N_PROBE_SAMPLES_FIM = 64
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Classical Model Hyperparameters
MLP_NW = 256  # Number of neurons in the second hidden layer

# Quantum Model Hyperparameters
L_QUBITS = 10
QNN_DEPTH = 10

# --- 2. Shared Data Preparation ---


def prepare_cifar100_binary_tasks():
    """
    Loads CIFAR-100, preprocesses it, and generates a list of binary classification tasks.
    This function is shared by both classical and quantum experiments.
    """
    print("1. Loading and preprocessing the complete CIFAR-100 dataset...")
    # TensorFlow can run on CPU for data loading
    with tf.device("/CPU:0"):
        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = (
            tf.keras.datasets.cifar100.load_data(label_mode="fine")
        )

    print("2. Converting color images to 32x32x1 grayscale...")
    x_train_gray = tf.image.rgb_to_grayscale(x_train_raw).numpy()
    x_test_gray = tf.image.rgb_to_grayscale(x_test_raw).numpy()

    print(
        f"3. Flattening images to {N_FEATURES} dimensions and L2-normalizing for amplitude encoding..."
    )
    x_train_flat = x_train_gray.reshape(-1, N_FEATURES).astype("float32") / 255.0
    x_test_flat = x_test_gray.reshape(-1, N_FEATURES).astype("float32") / 255.0

    def normalize(data):
        norm = np.linalg.norm(data, axis=1, keepdims=True)
        return data / (norm + 1e-9)

    x_train_norm = normalize(x_train_flat)
    x_test_norm = normalize(x_test_flat)

    y_train_labels = y_train_raw.flatten()
    y_test_labels = y_test_raw.flatten()

    print(f"4. Generating {NUM_TASKS} independent binary classification tasks...")
    tasks = []
    encoder = OneHotEncoder(sparse_output=False, categories=[range(N_CLASSES_PER_TASK)])

    for _ in range(NUM_TASKS):
        class_a, class_b = np.random.choice(100, 2, replace=False)

        train_indices = np.where(
            (y_train_labels == class_a) | (y_train_labels == class_b)
        )[0]
        y_train_task_remapped = (y_train_labels[train_indices] == class_b).astype(int)

        test_indices = np.where(
            (y_test_labels == class_a) | (y_test_labels == class_b)
        )[0]
        y_test_task_remapped = (y_test_labels[test_indices] == class_b).astype(int)

        tasks.append(
            {
                "train_x": x_train_norm[train_indices],
                "train_y_onehot": encoder.fit_transform(
                    y_train_task_remapped.reshape(-1, 1)
                ).astype(np.float32),
                "test_x": x_test_norm[test_indices],
                "test_y_labels": y_test_task_remapped,
            }
        )

    print("5. All task data prepared successfully!")
    return tasks


# --- 3. Model Definitions ---


class ClassicalClassifier:
    """Encapsulates the Keras MLP model, its training, evaluation, and FIM analysis."""

    def __init__(self, n_features, n_classes, lr, mlp_nw):
        self.model = self._build_model(n_features, n_classes, mlp_nw)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss_fn, metrics=["accuracy"]
        )
        print("Classical MLP model has been built and compiled.")
        self.model.summary()

    def _build_model(self, n_features, n_classes, mlp_nw):
        inputs = tf.keras.layers.Input(shape=(n_features,))
        x = tf.keras.layers.Dense(256, activation="relu")(inputs)
        x = tf.keras.layers.Dense(mlp_nw, activation="relu")(x)
        outputs = tf.keras.layers.Dense(n_classes)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def fit(self, xs_train, ys_train_onehot, epochs, batch_size):
        if len(xs_train) == 0:
            return
        self.model.fit(
            xs_train, ys_train_onehot, epochs=epochs, batch_size=batch_size, verbose=0
        )

    def evaluate(self, xs, ys_labels):
        if len(xs) == 0:
            return 0.0
        logits = self.model.predict(xs, batch_size=BATCH_SIZE, verbose=0)
        predictions = np.argmax(logits, axis=1)
        return np.mean(predictions == ys_labels)

    @tf.function
    def _calculate_grad_squared_norm_per_sample(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x[tf.newaxis, :], training=False)
            log_probs = tf.nn.log_softmax(logits)
            log_likelihood = tf.reduce_sum(log_probs * y)
        grads = tape.gradient(log_likelihood, self.model.trainable_variables)
        return tf.reduce_sum([tf.reduce_sum(g**2) for g in grads if g is not None])

    def get_fim_trace(self, x_probe, y_probe_onehot):
        if len(x_probe) == 0:
            return 0.0
        squared_norms = tf.data.Dataset.from_tensor_slices(
            (x_probe, y_probe_onehot)
        ).map(
            self._calculate_grad_squared_norm_per_sample,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        fim_trace = tf.reduce_mean(tf.stack(list(squared_norms)))
        return float(fim_trace.numpy())


class QuantumClassifier:
    """Encapsulates the JAX QNN model, its training, evaluation, and FIM analysis."""

    def __init__(self, L, d_main, n_classes, lr):
        self.K = tc.set_backend("jax")
        tc.set_dtype("complex64")
        self.L, self.d_main, self.n_classes, self.lr = L, d_main, n_classes, lr

        key = jax.random.PRNGKey(RANDOM_SEED)
        key_even, key_odd = jax.random.split(key)
        self.params = {
            "main_even": jax.random.uniform(
                key_even, shape=[d_main, L // 2, 15], maxval=2 * jnp.pi
            ),
            "main_odd": jax.random.uniform(
                key_odd, shape=[d_main, (L - 1) // 2, 15], maxval=2 * jnp.pi
            ),
        }
        self.setup_routines()

    def _build_ansatz(self, params, inputs):
        c = tc.Circuit(self.L, inputs=inputs)
        for d in range(self.d_main):
            for i in range(0, self.L, 2):
                if i + 1 < self.L:
                    c.su4(i, i + 1, theta=params["main_even"][d, i // 2])
            for i in range(1, self.L, 2):
                if i + 1 < self.L:
                    c.su4(i, i + 1, theta=params["main_odd"][d, i // 2])
        return c

    def _predict_logits(self, params, x_input):
        c = self._build_ansatz(params, x_input)
        # Use expectation of Z on the last qubit for binary classification
        exp_val = self.K.real(c.expectation_ps(z=[self.L - 1]))
        return jnp.array([exp_val, -exp_val])

    def setup_routines(self):
        self.batch_predict_logits = self.K.vmap(
            self._predict_logits, vectorized_argnums=1
        )

        def loss_fn(p, x, y):
            logits = self.batch_predict_logits(p, x)
            return jnp.mean(optax.softmax_cross_entropy(logits, y))

        def acc_fn(p, x, y):
            logits = self.batch_predict_logits(p, x)
            preds = jnp.argmax(logits, axis=1)
            return jnp.mean(preds == y)

        self.value_and_grad_fn = self.K.jit(self.K.value_and_grad(loss_fn))
        self.compute_accuracy = self.K.jit(acc_fn)
        self.optimizer = optax.adam(learning_rate=self.lr)
        self.opt_state = self.optimizer.init(self.params)

    def fit(self, xs_train, ys_train_onehot, epochs, batch_size):
        if len(xs_train) == 0:
            return
        dataset = tf.data.Dataset.from_tensor_slices((xs_train, ys_train_onehot))
        dataset = (
            dataset.shuffle(len(xs_train), seed=RANDOM_SEED)
            .batch(batch_size)
            .repeat(epochs)
        )

        for x_batch, y_batch in dataset:
            _, grads = self.value_and_grad_fn(
                self.params, x_batch.numpy(), y_batch.numpy()
            )
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.params = optax.apply_updates(self.params, updates)

    def evaluate(self, xs, ys_labels):
        if len(xs) == 0:
            return 0.0
        return float(self.compute_accuracy(self.params, xs, ys_labels))

    def get_fim_trace(self, x_probe, y_probe_onehot):
        if len(x_probe) == 0:
            return 0.0

        def log_likelihood(p, x, y):
            logits = self._predict_logits(p, x)
            log_probs = jax.nn.log_softmax(logits)
            return jnp.sum(log_probs * y)

        grad_fn = jax.grad(log_likelihood)
        batch_grads = jax.vmap(grad_fn, in_axes=(None, 0, 0))(
            self.params, x_probe, y_probe_onehot
        )

        def squared_norm(grad_tree):
            leaves = jax.tree_util.tree_leaves(grad_tree)
            return jnp.sum(jnp.array([jnp.sum(g**2) for g in leaves]))

        squared_norms = jax.vmap(squared_norm)(batch_grads)
        return float(jnp.mean(squared_norms))


# --- 4. Unified Experiment Runner ---


def run_experiment(model_type, tasks, x_probe, y_probe_onehot):
    """Runs the full continual learning experiment for a given model type."""
    print(f"\n{'='*20} Starting Experiment: {model_type.capitalize()} Model {'='*20}")

    if model_type == "classical":
        tf.config.set_visible_devices(
            tf.config.list_physical_devices("GPU"), "GPU"
        )  # Enable GPU for TF
        model = ClassicalClassifier(
            N_FEATURES, N_CLASSES_PER_TASK, LEARNING_RATE, MLP_NW
        )
    else:  # quantum
        tf.config.set_visible_devices(
            [], "GPU"
        )  # Disable GPU for TF to free memory for JAX
        model = QuantumClassifier(
            L_QUBITS, QNN_DEPTH, N_CLASSES_PER_TASK, LEARNING_RATE
        )

    results_history = {"accuracies": [], "fim_traces": []}

    for i in tqdm(range(NUM_TASKS), desc=f"Training {model_type.capitalize()} Model"):
        task_data = tasks[i]

        model.fit(
            task_data["train_x"],
            task_data["train_y_onehot"],
            EPOCHS_PER_TASK,
            BATCH_SIZE,
        )

        test_acc = model.evaluate(task_data["test_x"], task_data["test_y_labels"])
        results_history["accuracies"].append(test_acc)

        fim_trace = model.get_fim_trace(x_probe, y_probe_onehot)
        results_history["fim_traces"].append(fim_trace)

        if (i + 1) % 100 == 0:
            gc.collect()  # Periodically perform garbage collection

    print(f"{'='*20} Experiment Finished for {model_type.capitalize()} Model {'='*20}")

    for key in results_history:
        results_history[key] = np.array(results_history[key])

    return results_history


# --- 5. Main Execution Block ---

if __name__ == "__main__":
    total_start_time = time.time()

    # Prepare data once for both experiments
    cifar_tasks = prepare_cifar100_binary_tasks()

    # Create a fixed probe dataset for consistent FIM analysis across all tasks
    print("\nPreparing a fixed probe dataset for FIM analysis...")
    probe_indices = np.random.choice(
        len(cifar_tasks[0]["train_x"]), N_PROBE_SAMPLES_FIM, replace=False
    )
    x_probe_fim = cifar_tasks[0]["train_x"][probe_indices]
    y_probe_fim_onehot = cifar_tasks[0]["train_y_onehot"][probe_indices]
    print(f"FIM probe dataset created with size: {x_probe_fim.shape}")

    # Run classical experiment
    classical_results = run_experiment(
        "classical", cifar_tasks, x_probe_fim, y_probe_fim_onehot
    )
    np.savez("classical_cifar100_results.npz", **classical_results)
    print("Classical model results saved to 'classical_cifar100_results.npz'")

    # Run quantum experiment
    quantum_results = run_experiment(
        "quantum", cifar_tasks, x_probe_fim, y_probe_fim_onehot
    )
    np.savez("quantum_cifar100_results.npz", **quantum_results)
    print("Quantum model results saved to 'quantum_cifar100_results.npz'")

    total_end_time = time.time()
    print(
        f"\nAll experiments completed. Total time: {(total_end_time - total_start_time) / 60:.2f} minutes"
    )
