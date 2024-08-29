import jax
import jax.numpy as jnp

def create_instance_segmentation_mask(data, image_shape=(500, 500), num_classes=3, max_instances=4):
    def polygon_to_mask(shape, polygon):
        y, x = jnp.mgrid[:shape[0], :shape[1]]
        x, y = x.reshape(-1), y.reshape(-1)
        
        # Ensure polygon is at least 2D
        polygon = jnp.atleast_2d(jnp.array(polygon))
        n = polygon.shape[0]
        
        def body_fun(i, inside):
            j = (i + 1) % n
            p1x, p1y = polygon[i]
            p2x, p2y = polygon[j]
            
            mask = ((p1y > y) != (p2y > y)) & \
                   (x < (p2x - p1x) * (y - p1y) / (p2y - p1y + 1e-8) + p1x)
            return inside ^ mask
        
        inside = jax.lax.fori_loop(0, n, body_fun, jnp.zeros(x.shape[0], dtype=bool))
        return inside.reshape(shape)

    def process_single_instance(mask, instance_id, category, segmentation):
        # Ensure segmentation is at least 2D
        coords = jnp.atleast_2d(jnp.array(segmentation, dtype=jnp.int32))
        instance_mask = polygon_to_mask(image_shape, coords)
        return jnp.where(instance_mask, 
                         (category + 1) * (max_instances + 1) + instance_id, 
                         mask)

    mask = jnp.zeros(image_shape, dtype=jnp.int32)
    
    def body_fun(i, mask):
        category = jnp.array(data['category'])[i]
        segmentation = data['segmentation'][i]
        return process_single_instance(mask, i + 1, category, segmentation)

    num_instances = jnp.minimum(jnp.array(len(data['category']), dtype=jnp.int32), 
                                jnp.array(max_instances, dtype=jnp.int32))
    
    # Only process instances if there are any
    return jax.lax.cond(
        num_instances > 0,
        lambda x: jax.lax.fori_loop(0, x, body_fun, mask),
        lambda x: mask,
        num_instances
    )

# Example usage with various input formats:
data_1 = {
    "id": [1],
    "area": [1000],
    "bbox": [[100, 100, 50, 50]],
    "segmentation": [[[100, 100, 150, 100, 150, 150, 100, 150]]],
    "category": [0]
}

data_2 = {
    "id": [1],
    "area": [1000],
    "bbox": [[100, 100, 50, 50]],
    "segmentation": [[100, 100, 150, 100, 150, 150, 100, 150]],
    "category": [0]
}

mask_1 = create_instance_segmentation_mask(data_1)
mask_2 = create_instance_segmentation_mask(data_2)

print("Mask 1 shape:", mask_1.shape)
print("Mask 1 unique values:", jnp.unique(mask_1))
print("Mask 2 shape:", mask_2.shape)
print("Mask 2 unique values:", jnp.unique(mask_2))
