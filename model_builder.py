from tensorflow import keras

def build_model(architecture, search_space, input_shape=(28, 28, 1), num_classes=10):
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
    
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
