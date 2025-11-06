import tensorflow as tf
from tensorflow import keras


#for mnist

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

def conv3x3_approx(filters, mul_map_file='', use_bn=False):
    layers = []
    layers.append(FakeApproxConv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu', mul_map_file=mul_map_file))
    if use_bn:
        layers.append(keras.layers.BatchNormalization())
    return layers

def conv5x5_approx(filters, mul_map_file='', use_bn=False):
    layers = []
    layers.append(FakeApproxConv2D(filters=filters, kernel_size=(5, 5), padding='same', activation='relu', mul_map_file=mul_map_file))
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

def max_pool(filters):
    return [keras.layers.MaxPooling2D(2)]

def avg_pool(filters):
    return [keras.layers.AveragePooling2D(2)]

def get_search_space(use_approximate=False, mul_map_file=''):
    if use_approximate and APPROX_AVAILABLE:
        return {
            'conv3x3': lambda f, bn: conv3x3_approx(f, mul_map_file, bn),
            'conv5x5': lambda f, bn: conv5x5_approx(f, mul_map_file, bn),
            'max_pool': max_pool,
            'avg_pool': avg_pool,
        }
    else:
        return {
            'conv3x3': conv3x3_standard,
            'conv5x5': conv5x5_standard,
            'max_pool': max_pool,
            'avg_pool': avg_pool,
        }
        
        

