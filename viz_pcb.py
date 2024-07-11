import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state
import optax
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import os

# Import your model definition
from get_pcb import InstanceSegmentationModel, preprocess_batch

CKPT_DIR = '/Users/dougwoodward/dev/jets/pcb_checkpoints_o/'

def explore_directory(directory):
    print(f"Exploring directory: {directory}")
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

def load_checkpoint(checkpoint_dir):
    print(f"Attempting to load checkpoint from: {checkpoint_dir}")
    explore_directory(checkpoint_dir)

    # Create a checkpointer
    checkpointer = PyTreeCheckpointer()

    # Create a model instance
    model = InstanceSegmentationModel()

    # Create a dummy input to initialize the model
    dummy_input = jnp.ones((1, 480, 640, 3))  # Adjust shape as needed
    params = model.init(jax.random.PRNGKey(0), dummy_input)

    # Create an optimizer
    optimizer = optax.adam(learning_rate=1e-4)

    # Create an initial state
    initial_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

    # Create a CheckpointManager
    ckpt_manager = CheckpointManager(checkpoint_dir, checkpointer)

    try:
        # Attempt to restore the latest checkpoint
        step = ckpt_manager.latest_step()
        if step is None:
            raise ValueError("No checkpoints found in the directory.")
        restored_state = ckpt_manager.restore(step, items=initial_state)
        print(f"Checkpoint loaded successfully from step {step}.")
        return model, restored_state
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Please check the checkpoint directory structure and ensure it matches what was saved during training.")
        raise

def run_inference(model, state, image):
    # Ensure the image is in the correct format (add batch dimension if necessary)
    if image.ndim == 3:
        image = jnp.expand_dims(image, axis=0)

    # Run inference
    output = model.apply(state.params, image)
    return output

def visualize_segmentation(image, mask, output_file):
    # Convert JAX arrays to numpy if necessary
    if isinstance(image, jnp.ndarray):
        image = np.array(image)
    if isinstance(mask, jnp.ndarray):
        mask = np.array(mask)
    
    # Ensure image is in the correct range [0, 1]
    image = image.astype(np.float32) / 255.0 if image.max() > 1.0 else image
    
    # Create a color mask
    color_mask = np.zeros((*mask.shape[:2], 3), dtype=np.float32)
    color_mask[mask == 1] = [1, 0, 0]  # Red for class 1
    color_mask[mask == 2] = [0, 1, 0]  # Green for class 2
    # Add more colors for additional classes if needed
    
    # Create the overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(color_mask, alpha=0.5)
    plt.axis('off')
    plt.title("Segmentation Overlay")
    
    # Save the figure
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Visualization saved to {output_file}")

def main():
    # Load the checkpoint
    model, state = load_checkpoint(CKPT_DIR)

    # Load a test image
    test_dataset = load_dataset("keremberke/pcb-defect-segmentation", name="full", split="test")
    test_image = test_dataset[0]['image']
    
    # Preprocess the test image
    preprocessed_batch = preprocess_batch([{'image': test_image, 'objects': test_dataset[0]['objects']}])
    test_image = preprocessed_batch['images']

    # Run inference
    output = run_inference(model, state, test_image)

    # Process and display the output
    print("Output shape:", output.shape)
    print("Output:", output)

    # Assuming the output is a segmentation mask, we need to convert it to class labels
    # This step might need adjustment based on your model's exact output format
    segmentation_mask = jnp.argmax(output, axis=-1)

    # Visualize and save the segmentation
    visualize_segmentation(test_image[0], segmentation_mask[0], "segmentation_overlay.png")

if __name__ == "__main__":
    main()
