from operations import get_search_space
from architecture import random_architecture
from model_builder import build_model
from training import train_model, evaluate_model

def run_nas(x_train, y_train, x_val, y_val, num_trials=5, num_blocks=3, 
            use_approximate=False, mul_map_file='', logger=None,
            input_shape=(28, 28, 1), num_classes=10):  # ✅ Added parameters
    
    search_space = get_search_space(use_approximate=use_approximate, mul_map_file=mul_map_file)
    results = []
    
    mode = f"APPROXIMATE ({mul_map_file})" if use_approximate else "STANDARD"
    msg = f"Starting NAS: {mode}\n"
    if logger:
        logger.log(msg)
    else:
        print(msg)
    
    trial = 0
    attempts = 0
    max_attempts = num_trials * 3
    
    while trial < num_trials and attempts < max_attempts:
        attempts += 1
        
        architecture = random_architecture(search_space, num_blocks)
        
        consecutive_pools = 0
        max_consecutive_pools = 0
        for block in architecture:
            if 'pool' in block['op']:
                consecutive_pools += 1
                max_consecutive_pools = max(max_consecutive_pools, consecutive_pools)
            else:
                consecutive_pools = 0
        
        if max_consecutive_pools > 2:
            continue
        
        trial += 1
        
        try:
            model = build_model(
                architecture, 
                search_space,
                input_shape=input_shape,  # ✅ Pass shape
                num_classes=num_classes
            )
            train_model(model, x_train, y_train, x_val, y_val)
            accuracy = evaluate_model(model, x_val, y_val)
            
            if logger:
                logger.log_nas_trial(trial, num_trials, architecture, accuracy)
            else:
                print(f"Trial {trial}/{num_trials}")
                print(f"  {architecture}")
                print(f"  Accuracy: {accuracy:.4f}\n")
            
            results.append({
                'architecture': architecture,
                'accuracy': accuracy,
                'mode': mode
            })
        except Exception as e:
            msg = f"  ✗ Failed: {e}"
            if logger:
                logger.log(msg)
            else:
                print(msg)
            trial -= 1
            continue
    
    if not results:
        raise Exception("No valid architectures found!")
    
    best = max(results, key=lambda x: x['accuracy'])
    
    return results, best
