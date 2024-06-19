import pyarrow as pa
import tensorflow as tf
from datasets import load_dataset
import numpy as np
import jax.numpy as jnp
import jax

def get_dataset():
    # Load the HuggingFace dataset
    dataset = load_dataset("Illia56/Military-Aircraft-Detection")
    print(dataset)
    batch_size = 2
    # Access the individual datasets
    train_dataset = dataset["train"]
    # train_dataset = train_dataset.map(lambda x: tf.cast(x['image'], tf.float32))
    # eval_dataset = dataset["validation"]
    # Convert the HuggingFace dataset to a TensorFlow Dataset
    # train_dataset = tf.data.Dataset.from_tensor_slices(dataset["train"]["image"])
    #tf_train_dataset = train_dataset.to_tf_dataset(
    #    columns=["image"],
    #    # label_cols=["label_column"],
    #    shuffle=True,
    #    batch_size=batch_size,
    #)
    print("setup tf dataset")

    def preprocess_images(examples):
        # Decode and preprocess the images
        #print(examples)
        # images = [tf.cast(tf.io.decode_image(img, channels=3), tf.float32) for img in examples['image']]
        # images = [tf.io.decode_image(np.array(img).astype(np.float32), channels=3) for img in examples['image']]
        images = [np.array(img).astype(np.float32) for img in examples['image']]
        #print("----------")
        #print(images)
        #images = [tf.cast(img, tf.float32) for img in images]
        return {'image': images}
    # def preprocess_fn(x):
    #    return x
    # Preprocess the data
    train_dataset = train_dataset.map(preprocess_images, batched=True)

    # Batch the data
    #tf_train_dataset = tf_train_dataset.batch(batch_size)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    # Create a data loader for Flax
    train_data_loader = iter(train_dataset)
    
    return train_data_loader


def train_loop():
    train_data_loader = get_dataset() 
    print("Dataset acquired!")
    # Use the data loader in your Flax model
    print(train_data_loader)
    for batch in train_data_loader:
    #    # batch = jax.device_put(batch)
        print(batch)
        break
    # Train or evaluate your model with the batch

train_loop()
