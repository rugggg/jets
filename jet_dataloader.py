import pyarrow as pa
import tensorflow as tf
from datasets import load_dataset
import jax

# Load the HuggingFace dataset
dataset = load_dataset("Illia56/Military-Aircraft-Detection")
batch_size = 2
# Access the individual datasets
train_dataset = dataset["train"]
print(dataset.keys())
print(train_dataset)
# eval_dataset = dataset["validation"]
# Convert the HuggingFace dataset to a TensorFlow Dataset
tf_train_dataset = train_dataset.to_tf_dataset(
    columns=["image"],
    # label_cols=["label_column"],
    shuffle=True,
    batch_size=batch_size,
)


def preprocess_fn(x):
    return x
# Preprocess the data
#train_dataset = train_dataset.map(preprocess_fn)

# Batch the data
tf_train_dataset = tf_train_dataset.batch(batch_size)
# Create a data loader for Flax
train_data_loader = iter(tf_train_dataset)

# Use the data loader in your Flax model
for batch in train_data_loader:
    batch = jax.device_put(batch)
    # Train or evaluate your model with the batch
