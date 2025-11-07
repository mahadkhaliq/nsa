import tensorflow as tf
from tensorflow import keras

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

try:
    from keras.layers.fake_approx_convolutional import FakeApproxConv2D
    APPROX_AVAILABLE = True
    print("✓ Approximate layers loaded")
except Exception as e:
    print(f"⚠ Approximate layers not available: {e}")
    APPROX_AVAILABLE = False

# ============ EXISTING OPERATIONS ============
def conv3x3_approx(filters, mul_map_file='', use_bn=False):
    layers = []
    layers.append(FakeApproxConv2D(filters=filters, kernel_size=(3, 3), padding='same', 
                                   activation='relu', mul_map_file=mul_map_file))
    if use_bn:
        layers.append(keras.layers.BatchNormalization())
    return layers

def conv5x5_approx(filters, mul_map_file='', use_bn=False):
    layers = []
    layers.append(FakeApproxConv2D(filters=filters, kernel_size=(5, 5), padding='same', 
                                   activation='relu', mul_map_file=mul_map_file))
    if use_bn:
        layers.append(keras.layers.BatchNormalization())
    return layers

def conv3x3_standard(filters, use_bn=False):
    layers = []
    layers.append(keras.layers.Conv2D(filters, 3, padding='same', activation='relu'))
    if use_bn:
        layers.append(keras.layers.BatchNormalization())
    return layers

def conv5x5_standard(filters, use_bn=False):
    layers = []
    layers.append(keras.layers.Conv2D(filters, 5, padding='same', activation='relu'))
    if use_bn:
        layers.append(keras.layers.BatchNormalization())
    return layers

# ============ NEW OPERATIONS ============

# 1x1 Convolutions (channel mixing, very useful!)
def conv1x1_approx(filters, mul_map_file='', use_bn=False):
    layers = []
    layers.append(FakeApproxConv2D(filters=filters, kernel_size=(1, 1), padding='same', 
                                   activation='relu', mul_map_file=mul_map_file))
    if use_bn:
        layers.append(keras.layers.BatchNormalization())
    return layers

def conv1x1_standard(filters, use_bn=False):
    layers = []
    layers.append(keras.layers.Conv2D(filters, 1, padding='same', activation='relu'))
    if use_bn:
        layers.append(keras.layers.BatchNormalization())
    return layers

# # Depthwise Separable Convolutions (efficient!)
# def depthwise_sep_standard(filters, use_bn=False):
#     layers = []
#     layers.append(keras.layers.SeparableConv2D(filters, 3, padding='same', activation='relu'))
#     if use_bn:
#         layers.append(keras.layers.BatchNormalization())
#     return layers

# Identity (skip connection)
def identity(filters):
    return []  # No-op, maintains input

# Global Average Pooling (better than flatten for CNNs)
def global_avg_pool(filters):
    return [keras.layers.GlobalAveragePooling2D()]

def max_pool(filters):
    return [keras.layers.MaxPooling2D(2)]

def avg_pool(filters):
    return [keras.layers.AveragePooling2D(2)]

# Dropout layer (regularization)
def dropout(filters, rate=0.3):
    return [keras.layers.Dropout(rate)]

def get_search_space(use_approximate=False, mul_map_file='', include_advanced=True):
    """
    Get search space with optional advanced operations
    
    Args:
        use_approximate: Use approximate multipliers
        mul_map_file: Path to multiplier file
        include_advanced: Include advanced operations (depthwise, 1x1, etc)
    """
    if use_approximate and APPROX_AVAILABLE:
        base_space = {
            'conv3x3': lambda f, bn: conv3x3_approx(f, mul_map_file, bn),
            'conv5x5': lambda f, bn: conv5x5_approx(f, mul_map_file, bn),
            'max_pool': max_pool,
            'avg_pool': avg_pool,
        }
        
        if include_advanced:
            base_space.update({
                'conv1x1': lambda f, bn: conv1x1_approx(f, mul_map_file, bn),
                'identity': identity,
            })
    else:
        base_space = {
            'conv3x3': conv3x3_standard,
            'conv5x5': conv5x5_standard,
            'max_pool': max_pool,
            'avg_pool': avg_pool,
        }
        
        if include_advanced:
            base_space.update({
                'conv1x1': conv1x1_standard,
                # 'depthwise_sep': depthwise_sep_standard, causing problem for the multiplier
                'identity': identity,
            })
    
    return base_space