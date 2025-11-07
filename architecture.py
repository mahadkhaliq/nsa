import numpy as np

def random_architecture(search_space, num_blocks=3, filter_options=[32, 64, 128], 
                       max_pools=2, force_conv_start=True):
    """
    Generate random architecture with constraints
    
    Args:
        search_space: Available operations
        num_blocks: Number of blocks
        filter_options: Filter counts to choose from
        max_pools: Maximum number of pooling layers
        force_conv_start: Ensure first layer is convolution
    """
    operations = list(search_space.keys())
    architecture = []
    
    pool_count = 0
    consecutive_pools = 0
    
    for i in range(num_blocks):
        # First layer must be conv
        if i == 0 and force_conv_start:
            available_ops = [op for op in operations if 'conv' in op]
        # Limit pooling
        elif pool_count >= max_pools:
            available_ops = [op for op in operations if 'pool' not in op]
        # Avoid consecutive pools
        elif consecutive_pools >= 1:
            available_ops = [op for op in operations if 'pool' not in op]
        else:
            available_ops = operations
        
        op_name = np.random.choice(available_ops)
        
        # Progressive filter growth (common in CNNs)
        if i < num_blocks // 3:
            filters = np.random.choice(filter_options[:2])  # [32, 64]
        elif i < 2 * num_blocks // 3:
            filters = np.random.choice(filter_options[1:])  # [64, 128]
        else:
            filters = np.random.choice(filter_options)      # [32, 64, 128]
        
        use_bn = np.random.choice([True, False]) if 'conv' in op_name else False
        
        # Track pooling
        if 'pool' in op_name:
            pool_count += 1
            consecutive_pools += 1
        else:
            consecutive_pools = 0
        
        architecture.append({
            'op': op_name,
            'filters': filters,
            'use_bn': use_bn
        })
    
    return architecture

def mutate_architecture(architecture, search_space, mutation_rate=0.3):
    """
    Mutate an existing architecture for evolutionary search
    
    Args:
        architecture: Current architecture
        search_space: Available operations
        mutation_rate: Probability of mutating each block
    """
    new_arch = []
    operations = list(search_space.keys())
    filter_options = [32, 64, 128]
    
    for block in architecture:
        if np.random.random() < mutation_rate:
            # Mutate this block
            mutation_type = np.random.choice(['op', 'filters', 'bn'])
            
            if mutation_type == 'op':
                block['op'] = np.random.choice(operations)
            elif mutation_type == 'filters':
                block['filters'] = np.random.choice(filter_options)
            elif mutation_type == 'bn' and 'conv' in block['op']:
                block['use_bn'] = not block['use_bn']
        
        new_arch.append(block.copy())
    
    return new_arch