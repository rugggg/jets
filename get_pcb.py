import io
import numpy as np
from datasets import load_dataset
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
from flax.training import train_state
from jax import tree_util
import jax.scipy as jsp
from jax import lax
import orbax
import wandb
from tqdm import tqdm
from matplotlib.path import Path
from flax.training import checkpoints

CKPT_DIR = '/Users/dougwoodward/dev/jets/pcb_checkpoints_o/'
def get_data(split="train"):
    ds = load_dataset("keremberke/pcb-defect-segmentation", name="full")

    ds_jax = ds.with_format("jax")
    dataset = ds[split]
    dataset_jax = dataset.with_format("jax")
    return dataset_jax


class full_conv_encoder(nn.Module):
    
    latent_dim: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=8, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,2), strides=(2,2))

        x = nn.Conv(features=16, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,2), strides=(2,2))

        x = nn.Conv(features=32, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,2), strides=(2,2))

        x = x.reshape(x.shape[0], -1)  # Image grid to single feature vector
        x = nn.Dense(features=self.latent_dim)(x)

        return x


class full_conv_decoder(nn.Module):

    latent_dim: int = 128
    initial_shape: tuple[int, int] = (480, 640)
    num_encoder_layers: int = 3


    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=(self.initial_shape[0]//(2**self.num_encoder_layers))*(self.initial_shape[1]//(2**self.num_encoder_layers))*32)(x)
        x = x.reshape(-1, self.initial_shape[0]//(2**self.num_encoder_layers), self.initial_shape[1]//(2**self.num_encoder_layers), 32)
 
        x = nn.ConvTranspose(features=16, kernel_size=(3,3), strides=(2,2))(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=8, kernel_size=(3,3), strides=(2,2))(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=3, kernel_size=(3,3), strides=(2,2))(x)
        x = nn.tanh(x)

        return x


class SemanticSegmentationHead(nn.Module):

    num_classes: int = 3

    @nn.compact
    def __call__(self, x):
        # Instance Segmentation Head
        class_masks = nn.Conv(features=self.num_classes, kernel_size=(1, 1), name='instance_seg_head')(x)
        return class_masks


class InstanceSegmentationHead(nn.Module):

    num_instances: int = 7
    num_classes: int = 5

    @nn.compact
    def __call__(self, x):
        # Instance Segmentation Head
        instance_masks = nn.Conv(features=self.num_instances, kernel_size=(1, 1), name='instance_seg_head')(x)
        instance_masks = nn.sigmoid(instance_masks) # [batch_size, height, width, num_instances]
        class_logits = nn.Conv(self.num_instances * self.num_classes, kernel_size=(1, 1))(x)
        class_logits = class_logits.reshape(-1, self.num_instances, self.num_classes) # [batch_size, num_instances, num_classes]
        confidences = nn.Conv(features=self.num_instances, kernel_size=(1, 1), name='instance_confidences')(x)
        confidences = nn.sigmoid(confidences.mean(axis=(1,2))) # (batch_size, num_classes)
        return {"masks": instance_masks, "class_logits": class_logits, "confidences": confidences}


def rotate_point(point, angle, center):
    """Rotate a point around a given center."""
    angle_rad = np.radians(angle)
    cos_theta, sin_theta = np.cos(angle_rad), np.sin(angle_rad)
    
    x, y = point[0] - center[0], point[1] - center[1]
    new_x = x * cos_theta - y * sin_theta + center[0]
    new_y = x * sin_theta + y * cos_theta + center[1]
    
    return np.array([new_x, new_y])

def preprocess_data(json_data, image_shape=(480, 640)):
    """
    Convert JSON object detection data to a pixelwise mask.
    
    Args:
    json_data (dict): Object detection data with 'category' and 'segmentation' keys.
    image_shape (tuple): Shape of the output image (height, width).
    
    Returns:
    jnp.array: Pixelwise mask where each pixel value represents the object category.
    """
    mask = np.zeros(image_shape, dtype=np.int32)
    height, width = image_shape
    center = (width / 2, height / 2)  # Center of rotation

    for category, segmentation in zip(json_data['category'], json_data['segmentation']):
        # Convert segmentation to numpy array
        polygon = np.array(segmentation[0], dtype=np.int32).reshape((-1, 2))
        
        # Apply transformations to correct misalignment
        #polygon[:, 1] = image_shape[0] - polygon[:, 1]  # Vertical flip
        # Rotate polygon by 45 degrees
        # polygon = np.array([rotate_point(point, 90, center) for point in polygon])

        # You may need to adjust these scaling factors
        #polygon[:, 0] *= width / 100  # Scale X coordinates
        #polygon[:, 1] *= height / 100  # Scale Y coordinates
        
        polygon = polygon.astype(np.int32)
        # Create a mask for this polygon
        rr, cc = polygon_to_mask(polygon, image_shape)
        
        # Assign category to the masked area (adding 1 to reserve 0 for background)
        mask[rr, cc] = category + 1
    
    # Convert numpy array to jax array
    return jnp.array(mask)

def preprocess_batch(batch):
    images = [b['image'] for b in batch]
    json_data = [b['objects'] for b in batch]
    masks = np.stack([preprocess_data(data) for data in json_data])
    return {'images': jnp.array(images), 'masks': jnp.array(masks)}


def polygon_to_mask(polygon, shape):
    """
    Convert a polygon to a mask using the point-in-polygon algorithm.
    """
    mask = np.zeros(shape, dtype=bool)
    rr, cc = np.meshgrid(range(shape[0]), range(shape[1]), indexing='ij')
    rr, cc = rr.ravel(), cc.ravel()
    points = np.vstack((cc, rr)).T
    path = Path(polygon)
    mask_flat = path.contains_points(points)
    mask = mask_flat.reshape(shape)
    return np.where(mask)

@jax.jit
def hungarian_algorithm(cost_matrix):
    n, m = cost_matrix.shape
    
    # Step 1: Subtract row minima
    cost_matrix = cost_matrix - jnp.min(cost_matrix, axis=1, keepdims=True)
    
    # Step 2: Subtract column minima
    cost_matrix = cost_matrix - jnp.min(cost_matrix, axis=0, keepdims=True)
    
    # Step 3: Find a zero in each row
    row_ind = jnp.argmin(cost_matrix, axis=1)
    col_ind = jnp.arange(n)
    
    # Step 4: Create a mask for assignments
    mask = jnp.zeros_like(cost_matrix, dtype=bool)
    mask = mask.at[col_ind, row_ind].set(True)
    
    # Step 5: Greedy assignment for unassigned columns
    def assign_remaining(carry, x):
        mask, col_ind, row_ind = carry
        unassigned_rows = ~jnp.any(mask, axis=1)
        row = jnp.argmax(unassigned_rows)
        mask = mask.at[row, x].set(True)
        col_ind = col_ind.at[row].set(x)
        row_ind = row_ind.at[x].set(row)
        return (mask, col_ind, row_ind), None

    (mask, col_ind, row_ind), _ = jax.lax.scan(
        assign_remaining,
        (mask, col_ind, row_ind),
        jnp.arange(n)
    )
    
    return jnp.column_stack((col_ind, row_ind))


def segmentation_loss(predictions, targets, class_weights=None, epsilon=1e-7):
    """
    Compute pixel-wise segmentation loss for 3 classes.
    
    Args:
    predictions: JAX array of shape (batch_size, height, width, 3)
                 containing class probabilities for each pixel.
    targets: JAX array of shape (batch_size, height, width) containing 
             integer class labels (0, 1, or 2).
    class_weights: Optional JAX array of shape (3,) for class weighting.
    epsilon: Small constant to avoid log(0).
    
    Returns:
    Scalar loss value.
    """
    # Ensure predictions are probabilities
    predictions = jax.nn.softmax(predictions, axis=-1)
    targets = jnp.zeros((480, 640)) 
    # Convert targets to one-hot encoding
    targets_one_hot = jax.nn.one_hot(targets, 5)
    
    # Compute cross-entropy loss
    cross_entropy = -jnp.sum(targets_one_hot * jnp.log(predictions + epsilon), axis=-1)
    
    # Apply class weights if provided
    if class_weights is not None:
        weights = jnp.take(class_weights, targets)
        cross_entropy *= weights
    
    # Compute mean loss
    loss = jnp.mean(cross_entropy)
    
    return loss

def instance_segmentation_loss(predictions, targets, num_classes):
    masks_pred = predictions['masks']
    class_logits = jnp.expand_dims(predictions['class_logits'], 0)
    confidences = jnp.expand_dims(predictions['confidences'], 0)

    masks_true = targets['masks']
    classes_true = jnp.expand_dims(targets['classes'], 0)
    print(masks_pred.shape) 
    batch_size, height, width, num_instances = masks_pred.shape
    
    def single_image_loss(masks_pred, class_logits, confidences, masks_true, classes_true):
        print(masks_pred.shape)
        masks_pred = jnp.squeeze(masks_pred)
        masks_true = jnp.squeeze(masks_true)
        # Compute pairwise IoU between predicted and true masks
        intersection = jnp.sum(masks_pred[:, None] * masks_true[None, :], axis=(1, 2))
        union = jnp.sum(masks_pred[:, None] + masks_true[None, :], axis=(1, 2)) - intersection
        iou = intersection / (union + 1e-6)
        
        # Use negative IoU as cost for Hungarian matching
        cost_matrix = -iou
        
        indices = hungarian_algorithm(cost_matrix)
        print("indices shape", indices.shape)
        # Compute mask loss (Dice loss)
        matched_ious = iou[indices[:, 0], indices[:, 1]]
        mask_loss = 1 - (2 * matched_ious / (matched_ious + 1))
        
        # Compute classification loss
        matched_logits = class_logits[indices[:, 0]]
        matched_classes = classes_true[indices[:, 1]]
        print(matched_logits.shape, matched_classes.shape)
        class_loss = optax.softmax_cross_entropy(matched_logits, jax.nn.one_hot(matched_classes, num_classes))
        
        # Compute confidence loss (binary cross-entropy)
        conf_true = jnp.zeros_like(confidences)
        conf_true = conf_true.at[indices[:, 0]].set(1)
        conf_loss = jnp.mean(-(conf_true * jnp.log(confidences + 1e-6) + (1 - conf_true) * jnp.log(1 - confidences + 1e-6)))
        
        return jnp.mean(mask_loss) + jnp.mean(class_loss) + conf_loss
    
    # Vectorize the single image loss function over the batch
    batch_loss = jax.vmap(single_image_loss)
    print("Mask Pred Shape:", masks_pred.shape, "class logits.shape:", class_logits.shape, "confidences.shape:", confidences.shape, "true masks shape", masks_true.shape, "true classes.shape:", classes_true.shape) 
    total_loss = batch_loss(masks_pred, class_logits, confidences, masks_true, classes_true)
    return jnp.mean(total_loss)


def loss_fn(pred_instance_masks, targets):
    # instance_seg_loss = instance_segmentation_loss(pred_instance_masks, targets, num_classes=3) # 20 = num_classes?
    print(pred_instance_masks)
    sem_seg_loss = segmentation_loss(pred_instance_masks['masks'], targets, class_weights=None, epsilon=1e-7)
    total_loss = sem_seg_loss #instance_seg_loss # bbox_loss + instance_seg_loss + mask_loss
    return total_loss


class InstanceSegmentationModel(nn.Module):
    
    num_instances: int = 3
    num_classes: int = 5

    def setup(self, latent_dim: int = 128):
        self.encoder = full_conv_encoder(latent_dim)
        self.decoder = full_conv_decoder(latent_dim)

        self.semantic_seg_head = SemanticSegmentationHead(self.num_classes)
        #self.instance_seg_head = InstanceSegmentationHead(num_instances=self.num_instances, num_classes=self.num_classes)
        # self.out_layer = nn.Conv(features=1, kernel_size=(7, 7), strides=(2, 2), padding='SAME')


    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        
        #x_out = self.instance_seg_head(x_hat)
        x_out = self.semantic_seg_head(x_hat)
        return x_out



@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        print(batch)
        batch_im = batch['images'] # jnp.expand_dims(batch['image'], 0)
        predictions = state.apply_fn(params, batch_im)
        loss = segmentation_loss(predictions, batch['masks'])
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    # Update params using grads
    # This is a placeholder; replace with your actual update logic
    new_params = jax.tree.map(lambda p, g: p - 0.01 * g, state.params, grads)
    
    return state.replace(params=new_params), loss


def train_loop(initial_state, train_dataset, epochs=3):
    state = initial_state
    mgr_options = orbax.checkpoint.CheckpointManagerOptions(
            create=True,
            max_to_keep=3,
            keep_period=2,
            # step_prefix='pcb_checkpoint'
    )
    ckpt_mgr = orbax.checkpoint.CheckpointManager(
      CKPT_DIR,
      orbax.checkpoint.Checkpointer(
          orbax.checkpoint.PyTreeCheckpointHandler()
      ),
      mgr_options
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for raw_batch in tqdm(train_dataset):
            batch = preprocess_batch([raw_batch])
            state, loss = train_step(state, batch)
            epoch_loss += loss
            num_batches += 1
        save_args = flax.training.orbax_utils.save_args_from_target(state)
        ckpt_mgr.save(epoch, state, save_kwargs={'save_args': save_args})

        avg_loss = epoch_loss / num_batches
        wandb.log({"loss": avg_loss})
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")
    return state


if __name__ == "__main__":

    rng = jax.random.PRNGKey(42)

    num_epochs = 30
    lr = 1e-2

    train_dataset = get_data("train")
    # expand the sample image so it looks like batch of 1
    exp_img = jnp.expand_dims(train_dataset[0]["image"], 0)
    model = InstanceSegmentationModel()
    # init model
    # parameter init
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    print(model.tabulate(jax.random.key(0), exp_img))
    # Initialize the model
    params = model.init(init_rng, exp_img)
    optimizer = optax.adam(learning_rate=lr)
    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                            params=params,
                                            tx=optimizer)
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="pcb",
        # track hyperparameters and run metadata
        config={
        "architecture": "custom u-net",
        "dataset": "pcb-defect-segmentation",
        "learning_rate": lr,
        "epochs": num_epochs,
        }
    )


    out_params = train_loop(model_state, train_dataset, epochs=30)
    wandb.finish()
