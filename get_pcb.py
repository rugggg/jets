import io
from datasets import load_dataset
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
from flax.training import train_state


def get_data(split="train"):
    ds = load_dataset("keremberke/pcb-defect-segmentation", name="full")

    ds_jax = ds.with_format("jax")
    dataset = ds[split]
    dataset_jax = dataset.with_format("jax")
    return dataset_jax


class FPN(nn.Module):

    levels: int = 3
    
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Feature pyramid levels
        levels = []
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        prior_level = x
        for i in range(self.levels):
            x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
            l_out = nn.relu(x)
            x = nn.ConvTranspose(l_out, kernel_size=(3,), padding='VALID')
            x = jnp.concatenate(prior_level, axis=-1)
            prior_level = l_out
            levels.append(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        return x


class InstanceSegmentationModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = FPN()(x)
        x = nn.Conv(features=1, kernel_size=(7, 7), strides=(2, 2), padding='SAME')(x)
        return x


def iou_loss(y_pred, y_true):
    intersection = jnp.sum(y_pred * y_true, axis=(1, 2))
    union = jnp.sum(y_pred + y_true, axis=(1, 2))
    iou = intersection / union
    return 1 - iou

def calc_loss(state, params, batch):
    img = batch['image']
    objects = batch['segementation']
    pred = state.apply_fn(params, img)
    return iou_loss(pred, objects)

@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(iou_loss, argnums=1)
    optimizer = optax.adam(learning_rate=1e-4)
    y_pred = model.apply(params, x)
    loss = iou_loss(y_pred, y)
    grads = jax.grad(loss)(params)
    updates, params = optimizer.update(grads, params)
    return params

def train_loop(model, params, dataset, epochs=100):
    for epoch in range(100):
        for batch in dataset:
            print(batch)
            params = train_step(params, model, batch['image'], batch['segmentation'])   
    return params

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
    print(params)

    #  
    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                            params=params,
                                            tx=optimizer)


    out_params = train_loop(model, model_params, train_dataset, epochs=3)
