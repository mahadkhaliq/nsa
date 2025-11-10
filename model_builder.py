from tensorflow import keras
import numpy as np

def build_model(architecture, search_space, input_shape=(28, 28, 1), 
                num_classes=10, learning_rate=0.001, dropout_rate=0.5):
    """Build model with stronger regularization"""
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
    
    # Global pooling
    if x.shape[1] > 1:
        x = keras.layers.GlobalAveragePooling2D()(x)
    else:
        x = keras.layers.Flatten()(x)
    
    # Add stronger regularization
    x = keras.layers.Dropout(dropout_rate)(x)  # Increased from 0.3
    x = keras.layers.Dense(128, activation='relu', 
                          kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = keras.layers.Dropout(dropout_rate)(x)
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
