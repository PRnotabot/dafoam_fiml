#!/usr/bin/env python
"""
MLP training for cylinder FIML using TensorFlow.
Reads inverted betaFINuTilda field and trains NN to predict it from flow features.
"""

import numpy as np
from mpi4py import MPI
from pyofm import PYOFM
import tensorflow as tf
from tensorflow.keras import layers
import os

print(f"TensorFlow version: {tf.__version__}")
np.set_printoptions(precision=6, suppress=False)
np.random.seed(0)

# Initialize pyOFM for field I/O
ofm = PYOFM(comm=MPI.COMM_WORLD)

# Configuration
nCells = 50000  # Update with actual cell count from mesh
cases = ["c1_data"]  # Single case for cylinder
features = ["PoD", "VoS", "chiSA", "PSoSS"]

# Read training data
inputs = None
outputs = None

for case in cases:
    input_data = []
    output_data = []

    # Read input features
    for feature in features:
        field = np.zeros(nCells)
        ofm.readField(feature, "volScalarField", case, field)
        input_data.append(field)
    input_data = np.asarray(input_data).transpose()

    # Read output (inverted beta field)
    field = np.zeros(nCells)
    ofm.readField("betaFINuTilda", "volScalarField", case, field)
    output_data = field.reshape(-1, 1)

    if inputs is None:
        inputs = np.copy(input_data)
        outputs = np.copy(output_data)
    else:
        inputs = np.concatenate((inputs, input_data), axis=0)
        outputs = np.concatenate((outputs, output_data), axis=0)

print(f"Input shape: {inputs.shape}")
print(f"Output shape: {outputs.shape}")
print(f"Input range: [{inputs.min():.4f}, {inputs.max():.4f}]")
print(f"Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")

# Build MLP model with normalization layer
normalizer = layers.Normalization(input_shape=[len(features)], axis=None)
normalizer.adapt(inputs)

model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=50, activation="relu"),
    layers.Dense(units=50, activation="relu"),
    layers.Dense(units=1),
])

model.summary()

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss="mean_squared_error",
)

# Train
history = model.fit(
    inputs,
    outputs,
    epochs=500,
    batch_size=500,
    validation_split=0.2,
    verbose=1,
)

# Verify MSE
outputs_pred = model.predict(inputs, verbose=0)[:, 0]
mse = np.mean((outputs.flatten() - outputs_pred) ** 2)
print(f"Final MSE: {mse:.6f}")

# Save model
model.save("model")
print("Model saved to ./model")

# Write predicted beta field for debugging/visualization
ofm.writeField("betaFINuTildaNN", "volScalarField", outputs_pred[:nCells])
print("Predicted beta field written to betaFINuTildaNN")
