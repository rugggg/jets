import io

from seaborn.utils import axis_ticklabels_overlap
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

class InstanceSegmentationHead(nn.Module):

    num_instances: int = 7
    num_classes: int = 11

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


def instance_segmentation_loss(predictions, targets, num_classes):
    masks_pred = predictions['masks']
    class_logits = predictions['class_logits']
    confidences = predictions['confidences']

    masks_true = targets['masks']
    classes_true = targets['classes']
    
    batch_size, height, width, num_instances = masks_pred.shape
    
    def single_image_loss(masks_pred, class_logits, confidences, masks_true, classes_true):
        # Compute pairwise IoU between predicted and true masks
        intersection = jnp.sum(masks_pred[:, None] * masks_true[None, :], axis=(1, 2))
        union = jnp.sum(masks_pred[:, None] + masks_true[None, :], axis=(1, 2)) - intersection
        iou = intersection / (union + 1e-6)
        
        # Use negative IoU as cost for Hungarian matching
        cost_matrix = -iou
        
        # Perform Hungarian matching
        indices = jax.scipy.optimize.linear_sum_assignment(cost_matrix)
        indices = jnp.array(indices).T
        
        # Compute mask loss (Dice loss)
        matched_ious = iou[indices[:, 0], indices[:, 1]]
        mask_loss = 1 - (2 * matched_ious / (matched_ious + 1))
        
        # Compute classification loss
        matched_logits = class_logits[indices[:, 0]]
        matched_classes = classes_true[indices[:, 1]]
        class_loss = jax.nn.softmax_cross_entropy(matched_logits, jax.nn.one_hot(matched_classes, num_classes))
        
        # Compute confidence loss (binary cross-entropy)
        conf_true = jnp.zeros_like(confidences)
        conf_true = conf_true.at[indices[:, 0]].set(1)
        conf_loss = jnp.mean(-(conf_true * jnp.log(confidences + 1e-6) + (1 - conf_true) * jnp.log(1 - confidences + 1e-6)))
        
        return jnp.mean(mask_loss) + jnp.mean(class_loss) + conf_loss
    
    # Vectorize the single image loss function over the batch
    batch_loss = jax.vmap(single_image_loss)
    print(masks_pred.shape, class_logits.shape, confidences.shape, masks_true.shape, classes_true.shape) 
    total_loss = batch_loss(masks_pred, class_logits, confidences, masks_true, classes_true)
    return jnp.mean(total_loss)


def loss_fn(pred_instance_masks, targets):
    # bboxes, pred_instance_masks, pred_masks = model(inputs)
    # Compute losses
    #bbox_loss = ...  # Your bounding box loss implementation
    instance_seg_loss = instance_segmentation_loss(pred_instance_masks, targets, 7) # 20 = num_instances
    #mask_loss = ...  # Your segmentation mask loss implementation (optional)

    total_loss = instance_seg_loss # bbox_loss + instance_seg_loss + mask_loss
    return total_loss


class InstanceSegmentationModel(nn.Module):
    
    num_instances: int = 3
    num_classes: int = 4

    def setup(self, latent_dim: int = 128):
        self.encoder = full_conv_encoder(latent_dim)
        self.decoder = full_conv_decoder(latent_dim)
        self.instance_seg_head = InstanceSegmentationHead(num_instances=self.num_instances, num_classes=self.num_classes)
        # self.out_layer = nn.Conv(features=1, kernel_size=(7, 7), strides=(2, 2), padding='SAME')


    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        
        x_out = self.instance_seg_head(x_hat)
        return x_out


def convert_to_segmentation_format(instance_dict, mask_shape):
    classes = jnp.zeros(mask_shape[-1])
    mask = jnp.zeros(mask_shape)
    for idx, seg in enumerate(instance_dict["segmentation"]):
        category = instance_dict["category"][idx]
        mask.at[0:100, 0:100, idx].set(1)
        classes.at[idx].set(category)
        # segmentation = jnp.array(instance_dict["segmentation"][idx][0]).flatten()
    return {"masks": mask, "classes": classes}

def iou_loss(y_pred, y_true):
    intersection = jnp.sum(y_pred * y_true, axis=(1, 2))
    union = jnp.sum(y_pred + y_true, axis=(1, 2))
    iou = intersection / union
    return 1 - iou


@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(iou_loss, argnums=1)
    y_pred = state.apply_fn(state.params, jnp.expand_dims(batch['image'], 0)) # expand dims is hack for batch size 1
    loss = loss_fn(y_pred, convert_to_segmentation_format(batch['objects'], y_pred['masks'].shape))
    grads = jax.grad(loss)(state.params)
    updates, params = state.optimizer.update(grads, state.params)
    return params



def train_loop(model_state, dataset, epochs=100):

    for epoch in range(100):
        for batch in dataset:
            print("************")
            print(batch['image'].shape)
            model_state.params = train_step(model_state, batch)   
    return model_state

if __name__ == "__main__":

    rng = jax.random.PRNGKey(42)

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
    optimizer = optax.adam(learning_rate=1e-4)
    #  
    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                            params=params,
                                            tx=optimizer)


    out_params = train_loop(model_state, train_dataset, epochs=3)
