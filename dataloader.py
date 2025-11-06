import tensorflow as tf

def load_dataset(name, num_val=5000):
    """
    Generic loader for standard vision datasets.

    Args:
        name (str): Dataset name, e.g. 'mnist', 'cifar10', 'cifar100', 'fashion_mnist'.
        num_val (int): Number of validation samples to hold out from training set.

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test
    """
    dataset_loaders = {
        "mnist": tf.keras.datasets.mnist.load_data,
        "cifar10": tf.keras.datasets.cifar10.load_data,
        "cifar100": tf.keras.datasets.cifar100.load_data,
        "fashion_mnist": tf.keras.datasets.fashion_mnist.load_data
    }

    if name.lower() not in dataset_loaders:
        raise ValueError(f"Dataset {name} not supported.")

    (x_train, y_train), (x_test, y_test) = dataset_loaders[name.lower()]()

    # Preprocessing and flatten labels for compatibility
    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
    y_train, y_test = y_train.flatten(), y_test.flatten()

    # Normalization to [0, 1] for all except uint8 workflows
    if x_train.max() > 1.1:
        x_train /= 255.0
        x_test /= 255.0

    # Validation split
    x_val, y_val = x_train[-num_val:], y_train[-num_val:]
    x_train, y_train = x_train[:-num_val], y_train[:-num_val]

    return x_train, y_train, x_val, y_val, x_test, y_test

