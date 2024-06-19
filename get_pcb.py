import io
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

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.latent_dim)(x)
        x = nn.relu(x)

        x = x.reshape(x.shape[0], 4, 4, -1)
 
        x = nn.ConvTranspose(features=16, kernel_size=(3,3), strides=(2,2))(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=32, kernel_size=(3,3), strides=(2,2))(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=3, kernel_size=(3,3), strides=(2,2))(x)
        x = nn.tanh(x)

        return x

class InstanceSegmentationHead(nn.Module):

    num_instances: int = 10
    @nn.compact
    def __call__(self, x):
        # Instance Segmentation Head
        instance_masks = nn.Conv(features=self.num_instances, kernel_size=(1, 1), name='instance_seg_head')(x)
        instance_masks = nn.sigmoid(instance_masks) # [batch_size, height, width, num_instances]
        return instance_masks

def dice_loss(inputs, targets, smooth=1e-6):
    """Dice loss for binary segmentation"""
    inputs = inputs.reshape(-1)
    targets = targets.reshape(-1)
    intersection = jnp.sum(inputs * targets)
    union = jnp.sum(inputs) + jnp.sum(targets)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice

def instance_segmentation_loss(pred_masks, true_masks, num_instances):
    """Loss for instance segmentation head"""
    batch_size, height, width, _ = pred_masks.shape
    pred_masks = pred_masks.reshape(batch_size, height * width, num_instances)
    true_masks = true_masks.reshape(batch_size, height * width, num_instances)

    dice_losses = dice_loss(pred_masks, true_masks)
    dice_loss_sum = jnp.sum(dice_losses, axis=2)  # Sum across instances
    dice_loss_mean = jnp.mean(dice_loss_sum)  # Mean across batch and spatial dimensions

    return dice_loss_mean


def loss_fn(pred_instance_masks, targets):
    # bboxes, pred_instance_masks, pred_masks = model(inputs)

    # Compute losses
    #bbox_loss = ...  # Your bounding box loss implementation
    instance_seg_loss = instance_segmentation_loss(pred_instance_masks, targets, 20) # 20 = num_instances
    #mask_loss = ...  # Your segmentation mask loss implementation (optional)

    total_loss = instance_seg_loss # bbox_loss + instance_seg_loss + mask_loss
    return total_loss


class InstanceSegmentationModel(nn.Module):

    def setup(self, latent_dim: int = 128):
        self.encoder = full_conv_encoder(latent_dim)
        self.decoder = full_conv_decoder(latent_dim)
        self.instance_seg_head = InstanceSegmentationHead()
        # self.out_layer = nn.Conv(features=1, kernel_size=(7, 7), strides=(2, 2), padding='SAME')


    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        
        x_out = self.instance_seg_head(x_hat)
        return x_out


def convert_to_segmentation_format(instance_dict):
    # Extract relevant data from the dictionary
    category_id = instance_dict["category"][0]
    bbox = jnp.asarray(instance_dict["bbox"][0])
    segmentation = jnp.asarray(instance_dict["segmentation"][0])

    # Create a PyTree structure for the output
    output = {
        "category_id": category_id,
        "bbox": bbox,
        "segmentation": segmentation,
    }

    return tree_util.tree_flatten(output)[0]

def iou_loss(y_pred, y_true):
    print(y_true)
    intersection = jnp.sum(y_pred * y_true, axis=(1, 2))
    union = jnp.sum(y_pred + y_true, axis=(1, 2))
    iou = intersection / union
    return 1 - iou

def calc_loss(state, params, batch):
    img = batch['image']
    objects = batch['segementation']
    pred = state.apply_fn(params, img)
    print(batch['segmentation'])
    return loss_fn(pred, objects)

@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(iou_loss, argnums=1)
    y_pred = state.apply_fn(state.params, batch['image'])
    loss = loss_fn(y_pred, convert_to_segmentation_format(batch['objects']))
    grads = jax.grad(loss)(state.params)
    updates, params = state.optimizer.update(grads, state.params)
    return params

def train_loop(model_state, dataset, epochs=100):

    for epoch in range(100):
        for batch in dataset:
            print(batch['objects'])
            model_state.params = train_step(model_state, batch)   
    return model_state

if __name__ == "__main__":

    rng = jax.random.PRNGKey(42)

    train_dataset = get_data("train")
    exp_img = train_dataset[0]["image"]
    model = InstanceSegmentationModel()
    # init model
    # parameter init
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    # Initialize the model
    params = model.init(init_rng, exp_img)

    optimizer = optax.adam(learning_rate=1e-4)
    #  
    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                            params=params,
                                            tx=optimizer)


    out_params = train_loop(model_state, train_dataset, epochs=3)
