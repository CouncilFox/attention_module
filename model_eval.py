import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd


# Define the custom layers used in CBAM
class ChannelMean(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.keras.backend.mean(inputs, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)


class ChannelMax(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.keras.backend.max(inputs, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)


# Dictionary of custom layers
custom_objects = {
    "ChannelMean": ChannelMean,
    "ChannelMax": ChannelMax,
}

# Define model positions
model_positions = ["input", "pool1", "pool2", "pool3", "pool4", "pool5", "baseline"]

# Directory where models are saved
models_dir = "models"

# Set up data generator
val_dir = os.path.join("data", "imagenette2", "val")
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
)

# Initialize a list to store evaluation results
evaluation_results = []

# Load each model, evaluate, and store the results
for position in model_positions:
    if position != "baseline":
        model_filename = f"model_cbam_{position}.h5"
    else:
        model_filename = "model_cbam_baseline.h5"

    model_path = os.path.join(models_dir, model_filename)

    if os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        model = load_model(model_path, custom_objects=custom_objects)

        # Evaluate the model
        eval_loss, eval_accuracy = model.evaluate(validation_generator, verbose=0)
        print(
            f"Model: {position} | Loss: {eval_loss:.4f} | Accuracy: {eval_accuracy:.4f}"
        )

        # Append results to the list
        evaluation_results.append(
            {"Model_Position": position, "Loss": eval_loss, "Accuracy": eval_accuracy}
        )
    else:
        print(f"Model file {model_path} not found. Skipping {position}.")

# Convert the results list to a pandas DataFrame
results_df = pd.DataFrame(evaluation_results)

# Save the evaluation results to a CSV file
results_csv_path = os.path.join("evaluation_results", "model_evaluation.csv")
os.makedirs("evaluation_results", exist_ok=True)
results_df.to_csv(results_csv_path, index=False)
print(f"Evaluation results saved to {results_csv_path}")
