"""
Utility functions for the symbolic distillation pipeline.

Handles weight layout parsing, feature importance computation,
and knowledge distillation (teacher-to-student weight transfer)
for DAFoam neural network regression models.

Weight layout in DAFoam's flat parameter array (DARegression.C):
    For each hidden layer, for each neuron:
        - all input weights (from previous layer inputs)
        - 1 bias
    Then for the output neuron:
        - all weights from last hidden layer
        - 1 bias

Example: [4 inputs, [20, 20], 1 output]
    Layer 0: 20 neurons * (4 weights + 1 bias) = 100
    Layer 1: 20 neurons * (20 weights + 1 bias) = 420
    Output:  1 neuron  * (20 weights + 1 bias) = 21
    Total: 541 parameters
"""

import json
import numpy as np
from typing import Dict, List, Tuple


def compute_n_parameters(n_inputs: int, hidden_layers: List[int]) -> int:
    """Compute total parameter count matching DARegression::nParameters."""
    n = n_inputs * hidden_layers[0]  # input->hidden0 weights
    for i in range(1, len(hidden_layers)):
        n += hidden_layers[i] * hidden_layers[i - 1]  # hidden->hidden weights
    n += hidden_layers[-1] * 1  # hidden->output weights
    for h in hidden_layers:
        n += h  # hidden biases
    n += 1  # output bias
    return n


def parse_layer_params(parameters: np.ndarray, n_inputs: int, hidden_layers: List[int]):
    """
    Parse flat parameter array into per-layer weight matrices and bias vectors.

    Returns
    -------
    weights : list of np.ndarray
        weights[i] has shape (n_neurons_i, n_inputs_i)
    biases : list of np.ndarray
        biases[i] has shape (n_neurons_i,)
    """
    weights = []
    biases = []
    idx = 0

    for layer_i, n_neurons in enumerate(hidden_layers):
        n_in = n_inputs if layer_i == 0 else hidden_layers[layer_i - 1]
        W = np.zeros((n_neurons, n_in))
        b = np.zeros(n_neurons)
        for neuron_i in range(n_neurons):
            W[neuron_i, :] = parameters[idx : idx + n_in]
            idx += n_in
            b[neuron_i] = parameters[idx]
            idx += 1
        weights.append(W)
        biases.append(b)

    # Output layer
    n_last = hidden_layers[-1]
    W_out = np.zeros((1, n_last))
    W_out[0, :] = parameters[idx : idx + n_last]
    idx += n_last
    b_out = np.array([parameters[idx]])
    idx += 1

    weights.append(W_out)
    biases.append(b_out)

    return weights, biases


def flatten_layer_params(weights: list, biases: list) -> np.ndarray:
    """Flatten per-layer weights and biases back into DAFoam's flat parameter array."""
    params = []
    # Hidden layers
    for layer_i in range(len(weights) - 1):
        W = weights[layer_i]
        b = biases[layer_i]
        for neuron_i in range(W.shape[0]):
            params.extend(W[neuron_i, :].tolist())
            params.append(b[neuron_i])
    # Output layer
    W_out = weights[-1]
    b_out = biases[-1]
    params.extend(W_out[0, :].tolist())
    params.append(b_out[0])
    return np.array(params)


def compute_feature_importance(parameters: np.ndarray, n_inputs: int, hidden_layers: List[int]) -> np.ndarray:
    """
    Rank input features by mean absolute weight magnitude from the first hidden layer.

    Returns
    -------
    rankings : np.ndarray of shape (n_inputs,)
        Feature indices sorted by importance (most important first).
    """
    weights, _ = parse_layer_params(parameters, n_inputs, hidden_layers)
    W0 = weights[0]  # shape: (n_neurons_0, n_inputs)
    importance = np.mean(np.abs(W0), axis=0)  # mean over neurons for each input
    rankings = np.argsort(importance)[::-1]  # descending
    return rankings


def initialize_student_weights(
    teacher_params: np.ndarray,
    teacher_n_inputs: int,
    teacher_hidden: List[int],
    student_hidden: List[int],
    top_features: np.ndarray,
) -> np.ndarray:
    """
    Knowledge distillation: transfer teacher weights to a smaller student network.

    The student uses only a subset of input features (top_features) and
    may have a different (smaller) hidden architecture.

    Strategy:
        - First layer: extract the sub-matrix for top_features from teacher's W0.
          If student has fewer neurons, take the first N neurons (those with
          highest total weight magnitude).
        - Intermediate layers: if teacher has more layers than student, skip them.
          If shapes differ, truncate to the smaller dimension.
        - Output layer: truncate to match student's last hidden size.
        - All biases are copied where dimensions match, zero-padded otherwise.

    Parameters
    ----------
    teacher_params : np.ndarray
        Flat parameter array from trained teacher network.
    teacher_n_inputs : int
        Number of input features in teacher.
    teacher_hidden : list of int
        Teacher hidden layer sizes, e.g. [20, 20].
    student_hidden : list of int
        Student hidden layer sizes, e.g. [5].
    top_features : np.ndarray
        Indices of selected features (into teacher's input ordering).

    Returns
    -------
    np.ndarray
        Flat parameter array for the student network.
    """
    t_weights, t_biases = parse_layer_params(teacher_params, teacher_n_inputs, teacher_hidden)
    n_student_inputs = len(top_features)

    s_weights = []
    s_biases = []

    # --- First hidden layer ---
    t_W0 = t_weights[0]  # (teacher_neurons_0, teacher_n_inputs)
    t_b0 = t_biases[0]
    student_n0 = student_hidden[0]

    # Select columns for kept features
    t_W0_sub = t_W0[:, top_features]  # (teacher_neurons_0, n_student_inputs)

    # Rank teacher neurons by total weight magnitude (on kept features)
    neuron_importance = np.sum(np.abs(t_W0_sub), axis=1)
    top_neurons = np.argsort(neuron_importance)[::-1][:student_n0]
    top_neurons = np.sort(top_neurons)  # preserve ordering

    s_W0 = t_W0_sub[top_neurons, :]  # (student_n0, n_student_inputs)
    s_b0 = t_b0[top_neurons]
    s_weights.append(s_W0)
    s_biases.append(s_b0)

    prev_top_neurons = top_neurons

    # --- Intermediate hidden layers ---
    n_teacher_hidden = len(teacher_hidden)
    for s_layer_i in range(1, len(student_hidden)):
        student_ni = student_hidden[s_layer_i]

        if s_layer_i < n_teacher_hidden:
            # Teacher has a corresponding layer
            t_W = t_weights[s_layer_i]  # (teacher_neurons_i, teacher_neurons_{i-1})
            t_b = t_biases[s_layer_i]

            # Select rows/cols corresponding to kept neurons
            t_W_sub = t_W[:, prev_top_neurons]
            neuron_imp = np.sum(np.abs(t_W_sub), axis=1)
            top_n = np.argsort(neuron_imp)[::-1][:student_ni]
            top_n = np.sort(top_n)

            s_W = t_W_sub[top_n, :]
            s_b = t_b[top_n]
            prev_top_neurons = top_n
        else:
            # Student has more hidden layers than teacher; initialize randomly
            prev_size = student_hidden[s_layer_i - 1]
            s_W = (np.random.rand(student_ni, prev_size) - 0.5) * 0.02
            s_b = np.zeros(student_ni)

        s_weights.append(s_W)
        s_biases.append(s_b)

    # --- Output layer ---
    t_W_out = t_weights[-1]  # (1, teacher_last_hidden)
    t_b_out = t_biases[-1]

    if len(student_hidden) <= n_teacher_hidden:
        # Use weights corresponding to the kept neurons from the last used teacher layer
        s_W_out = t_W_out[:, prev_top_neurons]
    else:
        s_W_out = (np.random.rand(1, student_hidden[-1]) - 0.5) * 0.02

    s_weights.append(s_W_out)
    s_biases.append(t_b_out.copy())

    return flatten_layer_params(s_weights, s_biases)


def load_trained_parameters(json_path: str) -> np.ndarray:
    """Load designVariable.json and return the parameter1 array."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return np.array(data["parameter1"])


def save_parameters(parameters: np.ndarray, json_path: str):
    """Save parameters as designVariable JSON."""
    data = {"parameter1": parameters.tolist()}
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def evaluate_nn(parameters: np.ndarray, n_inputs: int, hidden_layers: List[int], X: np.ndarray) -> np.ndarray:
    """
    Evaluate the neural network on input array X.

    Parameters
    ----------
    parameters : np.ndarray
        Flat parameter array.
    n_inputs : int
        Number of input features.
    hidden_layers : list of int
        Hidden layer sizes.
    X : np.ndarray
        Input data, shape (n_samples, n_inputs).

    Returns
    -------
    np.ndarray
        Output values, shape (n_samples,).
    """
    weights, biases = parse_layer_params(parameters, n_inputs, hidden_layers)
    h = X  # (n_samples, n_inputs)
    for i in range(len(weights) - 1):
        h = h @ weights[i].T + biases[i]  # (n_samples, n_neurons_i)
        h = np.tanh(h)  # activation
    # Output layer (no activation)
    out = h @ weights[-1].T + biases[-1]  # (n_samples, 1)
    return out.flatten()
