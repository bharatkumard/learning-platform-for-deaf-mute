
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "isl_model.keras"
LABELS_PATH = BASE_DIR / "models" / "labels.npy"

# Compatibility workaround for older model configuration files.
original_from_config = keras.layers.Dense.from_config.__func__

def compatible_dense_from_config(cls, config):
    config = dict(config)
    config.pop("quantization_config", None)
    return original_from_config(cls, config)

keras.layers.Dense.from_config = classmethod(compatible_dense_from_config)

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("Loading ISL model...")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    labels = np.load(LABELS_PATH, allow_pickle=False)
    print("\nLabels:", labels)
    print("Number of labels:", len(labels))

except Exception as error:
    print("\nModel test failed:")
    print(type(error).__name__, "-", error)
