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
from PIL import Image

# Import your model definition
from get_pcb import InstanceSegmentationModel, preprocess_batch, preprocess_data

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

def visualize_masks(image, pred_mask, truth_mask, output_file):
    # Convert PIL image to numpy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert JAX arrays to numpy if necessary
    if isinstance(image, jnp.ndarray):
        image = np.array(image)
    if isinstance(pred_mask, jnp.ndarray):
        pred_mask = np.array(pred_mask)
    if isinstance(truth_mask, jnp.ndarray):
        truth_mask = np.array(truth_mask)
    
    # Ensure image is in the correct range [0, 1]
    image = image.astype(np.float32) / 255.0 if image.max() > 1.0 else image
    
    # Create color masks
    unique_classes = np.unique(np.concatenate((pred_mask, truth_mask)))
    cmap = plt.colormaps['tab10']
    colors = cmap(np.linspace(0, 1, len(unique_classes))) 

    def create_color_mask(mask):
        color_mask = np.zeros((*mask.shape[:2], 3), dtype=np.float32)
        for i, class_id in enumerate(unique_classes):
            if class_id == 0:  # Assuming 0 is background
                continue
            color_mask[mask == class_id] = colors[i, :3]
        return color_mask
    
    pred_color_mask = create_color_mask(pred_mask)
    truth_color_mask = create_color_mask(truth_mask)
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Predicted mask overlay
    ax1.imshow(image)
    ax1.imshow(pred_color_mask, alpha=0.5)
    ax1.axis('off')
    ax1.set_title("Predicted Segmentation")
    
    # Ground truth mask overlay
    ax2.imshow(image)
    ax2.imshow(truth_color_mask, alpha=0.5)
    ax2.axis('off')
    ax2.set_title("Ground Truth Segmentation")
    
    # Add colorbar
    unique_classes = sorted(unique_classes)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='tab10'), ax=[ax1, ax2], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_ticks(np.linspace(0, 1, len(unique_classes)))
    cbar.set_ticklabels(unique_classes)
    cbar.set_label('Class ID')
    
    # Save the figure
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Comparison visualization saved to {output_file}")

def main():
    # Load the checkpoint
    model, state = load_checkpoint(CKPT_DIR)

    # Load a test image
    test_dataset = load_dataset("keremberke/pcb-defect-segmentation", name="full", split="train")
    test_sample = test_dataset[100]
    
    # Get the original image
    test_image = test_sample['image']
    # Generate the ground truth mask
    ground_truth_mask = preprocess_data(test_sample['objects'])
    print(ground_truth_mask.max())
    print(ground_truth_mask[0])
    
    # Preprocess the test image for the model
    preprocessed_batch = preprocess_batch([{'image': test_image, 'objects': test_sample['objects']}])
    model_input_image = preprocessed_batch['images']

    # Run inference
    output = run_inference(model, state, model_input_image)
    
    # Convert model output to predicted mask
    # Adjust this based on your model's output format
    predicted_mask = jnp.argmax(output, axis=-1)[0]  # Assuming output is [batch, height, width, num_classes]

    # Visualize and save the comparison
    visualize_masks(test_image, predicted_mask, ground_truth_mask, "segmentation_comparison.png")

if __name__ == "__main__":
    main()

