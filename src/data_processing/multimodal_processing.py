import numpy as np
import tensorflow as tf

from sklearn.preprocessing import OrdinalEncoder

from collections import Counter


def decode_and_resize_from_path(
    path, h=224, w=224,
):
    bytes_ = tf.io.read_file(path)
    img = tf.image.decode_image(bytes_, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [h, w],
                          method=tf.image.ResizeMethod.LANCZOS5)
    img = tf.clip_by_value(img, 0.0, 255.0)
    img = tf.cast(img, tf.float64) / 255.0
    return img


def get_top_labels(labels, top_n=20):
    label_counts = Counter(labels)
    top_labels = [label for label, _ in label_counts.most_common(top_n)]
    print(f"Top {top_n} species (labels): {top_labels}")
    return set(top_labels)


def filter_by_labels_1(paths, labels, metadata, allowed_labels):
    # metadata must be a numpy array or df -> convert to numpy first
    
    # metadata_np = np.asarray(metadata, dtype=np.float64)

    filtered = []
    for i, (p, y) in enumerate(zip(paths, labels)):
        if y in allowed_labels:
            if i >= len(metadata):
                # print(f"Invalid row: {i}")
                continue
            else:
                # print(f"Valid row: {i}")
                m = metadata.iloc[i].to_numpy(dtype=np.float64)     # row i, shape (20,)
            filtered.append((p, m, y))

    f_paths, m_data, f_labels = zip(*filtered)
    return list(f_paths), np.array(m_data, dtype=np.float64), list(f_labels)


def filter_by_labels_2(paths, labels, metadata, allowed_labels):
    paths = np.asarray(paths)
    labels = np.asarray(labels)

    # Convert metadata only once, without copying if possible
    metadata_np = np.asarray(metadata, dtype=np.float64)

    allowed_labels = set(allowed_labels)

    mask = np.isin(labels, list(allowed_labels))

    return (
        paths[mask].tolist(),
        metadata_np[mask],        # view if possible
        labels[mask].tolist()
    )


def filter_by_labels_3(paths, labels, metadata_df, allowed_labels):
    mask = labels.isin(allowed_labels)

    return (
        paths[mask].tolist(),
        metadata_df.loc[mask].to_numpy(dtype=np.float64),
        labels[mask].tolist()
    )


def make_supervised_ds(paths, metadata, ref_labels, set_labels, training=True, shuffle_buf=10_000):
    """
    Creates a tf.data.Dataset:
      - reads (path, label)
      - decodes/resizes image
      - (re)maps label via label_map_tf to contiguous [0..n_classes-1]
      - optional flip augmentation
      - batches & prefetches
    """
    # Normalize metadata to prevent numerical instability
    metadata = np.array(metadata, dtype=np.float64)
    metadata_mean = metadata.mean(axis=0)
    metadata_std = metadata.std(axis=0)
    metadata_std[metadata_std == 0] = 1.0  # Avoid division by zero
    metadata = (metadata - metadata_mean) / metadata_std
    
    ds = tf.data.Dataset.from_tensor_slices((paths, metadata, set_labels))
    if training:
        ds = ds.shuffle(shuffle_buf, reshuffle_each_iteration=True)
    
    top_labels = sorted(list(ref_labels))
    label_map = {old: new for new, old in enumerate(top_labels)}

    label_map_tf = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(label_map.keys()),   dtype=tf.int32),
            values=tf.constant(list(label_map.values()), dtype=tf.int32),
        ),
        default_value=tf.constant(-1, dtype=tf.int32)
    )
    
    def _load_and_map(p, m, y):
        x = decode_and_resize_from_path(p)
        y = tf.cast(y, tf.int32)
        y = label_map_tf.lookup(y)

        tf.debugging.assert_greater_equal(y, tf.constant(0, tf.int32),
                                          message="Label not in label_map (got -1).")
        return x, m, tf.cast(y, tf.int32)

    autotune = tf.data.AUTOTUNE
    ds = ds.map(_load_and_map, num_parallel_calls=autotune)
    # ds = ds.map(_load_and_map_v2, num_parallel_calls=autotune)

    def _normalize(x, m, y):
        x = tf.image.convert_image_dtype(x, tf.float64)  # scales to [0,1]
        # Validate image values
        tf.debugging.assert_all_finite(x, message="Image contains NaN or Inf")
        # Validate metadata
        tf.debugging.assert_all_finite(m, message="Metadata contains NaN or Inf")
        return x, m, y

    ds = ds.map(_normalize, num_parallel_calls=autotune)

    def maybe_augment_image(x, m, y, augment_prob=0.1):
        # Draw a random number in [0,1)
        rand_val = tf.random.uniform([], 0, 1)

        def augment_fn():
            # Apply brightness or other augmentations here
            image_aug = tf.image.random_brightness(x, max_delta=0.2)
            return tf.clip_by_value(image_aug, 0.0, 1.0)

        # Apply augmentation with given probability
        return tf.cond(rand_val < augment_prob, augment_fn, lambda: x), m, y

    if training:
        ds = ds.map(lambda x, m, y: (tf.image.random_flip_left_right(x), m, y),
                    num_parallel_calls=autotune)
        ds = ds.map(maybe_augment_image, num_parallel_calls=autotune)

    def _pack_inputs(x, m, y):
        return {"image": x, "metadata": m}, y

    ds = ds.map(_pack_inputs)

    batch_size = 16
    ds = ds.batch(batch_size, drop_remainder=training).prefetch(autotune)

    opts = tf.data.Options(); opts.experimental_deterministic = False
    return ds.with_options(opts)