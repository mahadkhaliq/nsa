from tensorflow import keras
import numpy as np

try:
    from keras.layers.fake_approx_convolutional import FakeApproxConv2D
    APPROX_AVAILABLE = True
except ImportError:
    APPROX_AVAILABLE = False


def build_model_for_training(config, input_shape, num_classes, learning_rate=0.001):
    """
    Build model with STANDARD Conv2D for training.
    Matches reference implementation pattern.

    Args:
        config: dict with keys like 'conv1_filters', 'conv1_kernel', etc.
        input_shape: tuple (H, W, C)
        num_classes: int
        learning_rate: float
    """
    layers = []

    # Conv Block 1 - TWO consecutive convolutions
    layers.extend([
        keras.layers.Conv2D(
            filters=config['conv1_filters'],
            kernel_size=(config['conv1_kernel'], config['conv1_kernel']),
            padding='same',
            activation='relu',
            input_shape=input_shape
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(
            filters=config['conv1_filters'],
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        ),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2))
    ])

    # Conv Block 2 - TWO consecutive convolutions
    layers.extend([
        keras.layers.Conv2D(
            filters=config['conv2_filters'],
            kernel_size=(config['conv2_kernel'], config['conv2_kernel']),
            padding='same',
            activation='relu'
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(
            filters=config['conv2_filters'],
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        ),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2))
    ])

    # Conv Block 3 - Single conv + pooling
    layers.extend([
        keras.layers.Conv2D(
            filters=config['conv3_filters'],
            kernel_size=(config['conv3_kernel'], config['conv3_kernel']),
            padding='same',
            activation='relu'
        ),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2))
    ])

    # Dense layers with dropout
    layers.extend([
        keras.layers.Flatten(),
        keras.layers.Dense(config['dense1_size'], activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(config['dropout_rate']),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model = keras.Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_model_for_evaluation(config, input_shape, num_classes, learning_rate=0.001, weights=None):
    """
    Build model with FakeApproxConv2D for evaluation.
    CRITICAL: Only FIRST conv in each block uses approximate multiplier.
    SECOND conv is ALWAYS standard Conv2D for stability.

    Args:
        config: dict with multiplier files for each block
        input_shape: tuple (H, W, C)
        num_classes: int
        learning_rate: float
        weights: optional weights from trained standard model
    """
    if not APPROX_AVAILABLE:
        return build_model_for_training(config, input_shape, num_classes, learning_rate)

    layers = []

    # Conv Block 1
    # FIRST conv: Use approximate multiplier (if specified)
    if config['conv1_multiplier'] is None:
        layers.append(keras.layers.Conv2D(
            filters=config['conv1_filters'],
            kernel_size=(config['conv1_kernel'], config['conv1_kernel']),
            padding='same',
            activation='relu',
            input_shape=input_shape
        ))
    else:
        layers.append(FakeApproxConv2D(
            filters=config['conv1_filters'],
            kernel_size=(config['conv1_kernel'], config['conv1_kernel']),
            padding='same',
            activation='relu',
            mul_map_file=config['conv1_multiplier'],
            input_shape=input_shape
        ))

    # SECOND conv: ALWAYS standard Conv2D
    layers.extend([
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(
            filters=config['conv1_filters'],
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        ),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2))
    ])

    # Conv Block 2
    # FIRST conv: Use approximate multiplier (if specified)
    if config['conv2_multiplier'] is None:
        layers.append(keras.layers.Conv2D(
            filters=config['conv2_filters'],
            kernel_size=(config['conv2_kernel'], config['conv2_kernel']),
            padding='same',
            activation='relu'
        ))
    else:
        layers.append(FakeApproxConv2D(
            filters=config['conv2_filters'],
            kernel_size=(config['conv2_kernel'], config['conv2_kernel']),
            padding='same',
            activation='relu',
            mul_map_file=config['conv2_multiplier']
        ))

    # SECOND conv: ALWAYS standard Conv2D
    layers.extend([
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(
            filters=config['conv2_filters'],
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        ),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2))
    ])

    # Conv Block 3
    # Use approximate multiplier (if specified)
    if config['conv3_multiplier'] is None:
        layers.append(keras.layers.Conv2D(
            filters=config['conv3_filters'],
            kernel_size=(config['conv3_kernel'], config['conv3_kernel']),
            padding='same',
            activation='relu'
        ))
    else:
        layers.append(FakeApproxConv2D(
            filters=config['conv3_filters'],
            kernel_size=(config['conv3_kernel'], config['conv3_kernel']),
            padding='same',
            activation='relu',
            mul_map_file=config['conv3_multiplier']
        ))

    layers.extend([
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2))
    ])

    # Dense layers
    layers.extend([
        keras.layers.Flatten(),
        keras.layers.Dense(config['dense1_size'], activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(config['dropout_rate']),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model = keras.Sequential(layers)

    # Transfer weights if provided
    if weights is not None:
        model.set_weights(weights)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_model(architecture, search_space, input_shape=(28, 28, 1),
                num_classes=10, learning_rate=0.001, dropout_rate=0.5):
    """
    Legacy model builder - kept for backward compatibility with existing tests.
    For NAS, use build_model_for_training and build_model_for_evaluation instead.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs

    for block in architecture:
        op_name = block['op']
        filters = block['filters']
        use_bn = block.get('use_bn', False)

        operation = search_space[op_name]

        if 'conv' in op_name:
            layers_list = operation(filters, use_bn)
        else:
            layers_list = operation(filters)

        for layer in layers_list:
            x = layer(x)

    # Simple classifier head
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def count_model_params(model):
    """Count trainable parameters"""
    return int(np.sum([np.prod(v.get_shape()) for v in model.trainable_weights]))

def estimate_flops(architecture, input_shape):
    """Estimate FLOPs for architecture"""
    flops = 0
    h, w, c = input_shape

    for block in architecture:
        if 'conv3x3' in block['op']:
            flops += h * w * c * block['filters'] * 9
        elif 'conv5x5' in block['op']:
            flops += h * w * c * block['filters'] * 25
        elif 'conv1x1' in block['op']:
            flops += h * w * c * block['filters']
        elif 'pool' in block['op']:
            h, w = h//2, w//2

        c = block['filters']

    return flops
