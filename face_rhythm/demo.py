import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import resort_rows_hclust
from coupled_model import CoupledFaceRhythm

datapath = "/home/alex/data/face_rhythm/"


# Load face tensor and reshape to (face_pts, frequencies, timebins)
face_tensor = np.load(os.path.join(datapath, "Sxx_withinWS.npy"))
face_tensor = np.concatenate(
    (face_tensor[:, :, :, 0], face_tensor[:, :, :, 1]), axis=-1
).T

# Load neural time series in (neurons, timebins)
neural_matrix = np.load(
    os.path.join(datapath, "S2p", "F.npy")
).astype("float64")

# Normalize neural data.
neural_matrix -= np.min(neural_matrix, axis=1, keepdims=True)
neural_matrix /= np.std(neural_matrix, axis=1, keepdims=True)
# neural_matrix /= np.percentile(neural_matrix, 90, axis=1, keepdims=True)

# # Sort neurons.
# neural_matrix = neural_matrix[resort_rows_hclust(neural_matrix)]

plt.imshow(neural_matrix, aspect="auto")
plt.show()

# model = CoupledFaceRhythm(
#     num_components, num_neurons, num_neural_timebins,
#     neuron_window_size, num_facepts, num_freq, num_face_timebins,
#     init_matrix_norm, init_tensor_norm
# )

# face_est, neural_est = model()
