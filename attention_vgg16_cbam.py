# evaluate_and_visualize.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import tensorflow as tf


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

# Directory where models are saved
models_dir = "models"

# Directory to save evaluation results
results_dir = "evaluation_results"
os.makedirs(results_dir, exist_ok=True)

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

# Define the WordNet ID to class name mapping (update if necessary)
wnid_to_class = {
    "n01440764": "Tench",
    "n02099267": "English Springer Spaniel",
    "n04467665": "Cassette Player",
    "n03000684": "Chain Saw",
    "n09421951": "Church",
    "n03485407": "French Horn",
    "n03770679": "Garbage Truck",
    "n03594945": "Gas Pump",
    "n03255030": "Golf Ball",
    "n02823428": "Parachute",
}

# Retrieve class indices and create a mapping from index to WordNet ID
class_indices = validation_generator.class_indices
idx_to_wnid = {v: k for k, v in class_indices.items()}

# Determine if class names are WordNet IDs or human-readable
sample_class_name = list(validation_generator.class_indices.keys())[0]
if sample_class_name.startswith("n0"):
    print("Detected class names as WordNet IDs. Using wnid_to_class mapping.")
    # Create a mapping from index to class name using wnid_to_class
    idx_to_class = {
        idx: wnid_to_class.get(wnid, "Unknown") for idx, wnid in idx_to_wnid.items()
    }
else:
    print("Detected class names as human-readable. Using direct mapping.")
    # Create a mapping from index to class name, replacing underscores with spaces
    idx_to_class = {
        idx: class_name.replace("_", " ") for idx, class_name in idx_to_wnid.items()
    }

# Verify that all classes are mapped correctly
unknown_classes = [
    idx for idx, class_name in idx_to_class.items() if class_name == "Unknown"
]
if unknown_classes:
    print(
        f"Warning: The following class indices are mapped to 'Unknown': {unknown_classes}"
    )
else:
    print("All class indices successfully mapped to class names.")

# Initialize a list to store evaluation results
evaluation_results = []


# Function to evaluate a single model
def evaluate_model(model_path, position):
    model = load_model(model_path, custom_objects=custom_objects)
    loss, accuracy = model.evaluate(validation_generator, verbose=0)
    print(f"Model Position: {position} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")
    return {"Model_Position": position, "Loss": loss, "Accuracy": accuracy}


# Iterate through all models in the models directory
for filename in os.listdir(models_dir):
    if filename.endswith(".h5"):
        position = filename.replace("model_cbam_", "").replace(".h5", "")
        model_path = os.path.join(models_dir, filename)
        result = evaluate_model(model_path, position)
        evaluation_results.append(result)

# Convert the results list to a pandas DataFrame
results_df = pd.DataFrame(evaluation_results)

# Save the evaluation results to a CSV file
results_csv_path = os.path.join(results_dir, "model_evaluation.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"Evaluation results saved to {results_csv_path}")

# Plotting Accuracy and Loss
sns.set(style="whitegrid")

# Plot Accuracy
plt.figure(figsize=(10, 6))
sns.barplot(x="Model_Position", y="Accuracy", data=results_df, palette="viridis")
plt.title("Model Accuracy Comparison")
plt.xlabel("CBAM Insertion Position")
plt.ylabel("Accuracy")
plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
plt.tight_layout()

# Save the accuracy plot
accuracy_plot_path = os.path.join(results_dir, "model_accuracy_comparison.png")
plt.savefig(accuracy_plot_path)
plt.show()
print(f"Accuracy comparison plot saved to {accuracy_plot_path}")

# Plot Loss
plt.figure(figsize=(10, 6))
sns.barplot(x="Model_Position", y="Loss", data=results_df, palette="magma")
plt.title("Model Loss Comparison")
plt.xlabel("CBAM Insertion Position")
plt.ylabel("Loss")
plt.tight_layout()

# Save the loss plot
loss_plot_path = os.path.join(results_dir, "model_loss_comparison.png")
plt.savefig(loss_plot_path)
plt.show()
print(f"Loss comparison plot saved to {loss_plot_path}")

# Combined Accuracy and Loss Plot
melted_df = results_df.melt(
    id_vars=["Model_Position"],
    value_vars=["Accuracy", "Loss"],
    var_name="Metric",
    value_name="Value",
)

plt.figure(figsize=(12, 8))
sns.barplot(x="Model_Position", y="Value", hue="Metric", data=melted_df, palette="Set2")
plt.title("Model Accuracy and Loss Comparison")
plt.xlabel("CBAM Insertion Position")
plt.ylabel("Value")
plt.ylim(0, max(melted_df["Value"]) * 1.1)  # Add some space on top
plt.legend(title="Metric")
plt.tight_layout()

# Save the combined plot
combined_plot_path = os.path.join(results_dir, "model_accuracy_loss_comparison.png")
plt.savefig(combined_plot_path)
plt.show()
print(f"Combined accuracy and loss comparison plot saved to {combined_plot_path}")
