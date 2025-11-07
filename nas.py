from operations import get_search_space
from architecture import random_architecture, mutate_architecture
from model_builder import build_model
from training import train_model, evaluate_model
import numpy as np

def run_nas(x_train, y_train, x_val, y_val, num_trials=20, num_blocks=3, 
            use_approximate=False, mul_map_file='', logger=None,
            input_shape=(28, 28, 1), num_classes=10, method='evolutionary'):
    """
    Run NAS with evolutionary or random search
    
    Args:
        method: 'random' or 'evolutionary'
    """
    
    search_space = get_search_space(use_approximate=use_approximate, 
                                    mul_map_file=mul_map_file,
                                    include_advanced=True)
    results = []
    
    mode = f"APPROXIMATE ({mul_map_file})" if use_approximate else "STANDARD"
    msg = f"Starting NAS: {mode} - Method: {method.upper()}\n"
    if logger:
        logger.log(msg)
    else:
        print(msg)
    
    if method == 'evolutionary':
        return evolutionary_nas(x_train, y_train, x_val, y_val, num_trials, num_blocks,
                               search_space, logger, input_shape, num_classes, mode)
    else:
        return random_nas(x_train, y_train, x_val, y_val, num_trials, num_blocks,
                         search_space, logger, input_shape, num_classes, mode)

def random_nas(x_train, y_train, x_val, y_val, num_trials, num_blocks,
               search_space, logger, input_shape, num_classes, mode):
    """Original random search"""
    results = []
    trial = 0
    attempts = 0
    max_attempts = num_trials * 3
    
    while trial < num_trials and attempts < max_attempts:
        attempts += 1
        architecture = random_architecture(search_space, num_blocks)
        
        # Validation
        if not validate_architecture(architecture):
            continue
        
        trial += 1
        
        try:
            model = build_model(architecture, search_space, input_shape, num_classes)
            train_model(model, x_train, y_train, x_val, y_val)
            accuracy = evaluate_model(model, x_val, y_val)
            
            if logger:
                logger.log_nas_trial(trial, num_trials, architecture, accuracy)
            
            results.append({
                'architecture': architecture,
                'accuracy': accuracy,
                'mode': mode
            })
        except Exception as e:
            if logger:
                logger.log(f"  ✗ Failed: {e}")
            trial -= 1
    
    if not results:
        raise Exception("No valid architectures found!")
    
    best = max(results, key=lambda x: x['accuracy'])
    return results, best

def evolutionary_nas(x_train, y_train, x_val, y_val, num_trials, num_blocks,
                    search_space, logger, input_shape, num_classes, mode):
    """Evolutionary search with mutation"""
    results = []
    population = []
    
    # Initial population (20% of trials)
    init_size = max(3, num_trials // 5)
    
    if logger:
        logger.log(f"Phase 1: Generating initial population ({init_size} architectures)")
    
    for i in range(init_size):
        architecture = random_architecture(search_space, num_blocks)
        
        if not validate_architecture(architecture):
            continue
        
        try:
            model = build_model(architecture, search_space, input_shape, num_classes)
            train_model(model, x_train, y_train, x_val, y_val, epochs=5)  # Quick eval
            accuracy = evaluate_model(model, x_val, y_val)
            
            population.append({
                'architecture': architecture,
                'accuracy': accuracy
            })
            
            if logger:
                logger.log(f"Init {i+1}/{init_size}: Accuracy {accuracy:.4f}")
        except:
            continue
    
    # Evolution phase
    if logger:
        logger.log(f"\nPhase 2: Evolution ({num_trials - init_size} generations)")
    
    for trial in range(init_size, num_trials):
        # Select top 50% as parents
        population.sort(key=lambda x: x['accuracy'], reverse=True)
        parents = population[:len(population)//2]
        
        # Select random parent and mutate
        parent = np.random.choice(parents)
        architecture = mutate_architecture(parent['architecture'], search_space)
        
        if not validate_architecture(architecture):
            continue
        
        try:
            model = build_model(architecture, search_space, input_shape, num_classes)
            train_model(model, x_train, y_train, x_val, y_val)
            accuracy = evaluate_model(model, x_val, y_val)
            
            population.append({
                'architecture': architecture,
                'accuracy': accuracy
            })
            
            results.append({
                'architecture': architecture,
                'accuracy': accuracy,
                'mode': mode
            })
            
            if logger:
                logger.log_nas_trial(trial+1, num_trials, architecture, accuracy)
            
            # Keep population size manageable
            if len(population) > 10:
                population = population[:10]
                
        except Exception as e:
            if logger:
                logger.log(f"  ✗ Failed: {e}")
    
    if not results:
        raise Exception("No valid architectures found!")
    
    best = max(results, key=lambda x: x['accuracy'])
    return results, best

def validate_architecture(architecture):
    """Check if architecture is valid"""
    consecutive_pools = 0
    pool_count = 0
    
    for block in architecture:
        if 'pool' in block['op']:
            consecutive_pools += 1
            pool_count += 1
            if consecutive_pools > 2 or pool_count > 3:
                return False
        else:
            consecutive_pools = 0
    
    return True