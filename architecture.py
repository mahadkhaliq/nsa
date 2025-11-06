import numpy as np

def random_architecture(search_space, num_blocks=3, filter_options=[32, 64, 128]):
    operations = list(search_space.keys())
    architecture = []
    
    for _ in range(num_blocks):
        op_name = np.random.choice(operations)
        filters = np.random.choice(filter_options)
        use_bn = np.random.choice([True, False]) if 'conv' in op_name else False
        
        architecture.append({
            'op': op_name,
            'filters': filters,
            'use_bn': use_bn
        })
    
    return architecture