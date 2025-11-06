import numpy as np
import os

def analyze_multiplier_bin(bin_file):
    """Analyze a multiplier .bin file to extract error characteristics"""
    
    if not os.path.exists(bin_file):
        print(f"File not found: {bin_file}")
        return
    
    print("="*70)
    print(f"MULTIPLIER ANALYSIS: {os.path.basename(bin_file)}")
    print("="*70)
    
    # Read the binary multiplication table
    with open(bin_file, 'rb') as f:
        data = f.read()
    
    # Convert to numpy array (8-bit multiplication table is 256x256 = 65536 entries)
    if len(data) == 65536:
        mul_table = np.frombuffer(data, dtype=np.uint8)
        mul_table = mul_table.reshape(256, 256)
        print(f"✓ Loaded 8-bit unsigned multiplier table (256x256)")
    elif len(data) == 65536 * 2:  # 16-bit results
        mul_table = np.frombuffer(data, dtype=np.uint16)
        mul_table = mul_table.reshape(256, 256)
        print(f"✓ Loaded 8-bit unsigned multiplier table with 16-bit results (256x256)")
    else:
        print(f"✗ Unexpected file size: {len(data)} bytes")
        return
    
    # Calculate exact multiplication table
    exact_table = np.zeros((256, 256), dtype=np.uint16)
    for i in range(256):
        for j in range(256):
            exact_table[i, j] = i * j
    
    # Calculate error metrics
    errors = mul_table.astype(np.int32) - exact_table.astype(np.int32)
    abs_errors = np.abs(errors)
    
    # Error metrics
    mae = np.mean(abs_errors)  # Mean Absolute Error
    mse = np.mean(errors ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    med = np.median(abs_errors)  # Median Error Distance
    wce = np.max(abs_errors)  # Worst Case Error
    mred = np.mean(np.abs(errors / (exact_table + 1e-10)))  # Mean Relative Error
    
    # Error distribution
    zero_errors = np.sum(abs_errors == 0)
    small_errors = np.sum(abs_errors <= 5)
    medium_errors = np.sum((abs_errors > 5) & (abs_errors <= 20))
    large_errors = np.sum(abs_errors > 20)
    
    # Hardware estimates (relative to exact multiplier)
    # These are rough estimates based on typical approximate multiplier characteristics
    error_rate = 1.0 - (zero_errors / 65536)
    power_saving = min(0.5, error_rate * 0.6)  # Rough estimate
    area_saving = min(0.4, error_rate * 0.5)   # Rough estimate
    
    print(f"\n{'='*70}")
    print("ERROR METRICS")
    print("="*70)
    print(f"Mean Absolute Error (MAE):     {mae:.4f}")
    print(f"Mean Squared Error (MSE):      {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Median Error Distance (MED):   {med:.4f}")
    print(f"Worst Case Error (WCE):        {wce}")
    print(f"Mean Relative Error (MRED):    {mred*100:.2f}%")
    
    print(f"\n{'='*70}")
    print("ERROR DISTRIBUTION")
    print("="*70)
    print(f"Exact matches (error = 0):     {zero_errors:,} / 65,536 ({zero_errors/65536*100:.2f}%)")
    print(f"Small errors (≤5):             {small_errors:,} / 65,536 ({small_errors/65536*100:.2f}%)")
    print(f"Medium errors (6-20):          {medium_errors:,} / 65,536 ({medium_errors/65536*100:.2f}%)")
    print(f"Large errors (>20):            {large_errors:,} / 65,536 ({large_errors/65536*100:.2f}%)")
    
    print(f"\n{'='*70}")
    print("ESTIMATED HARDWARE BENEFITS")
    print("="*70)
    print(f"Estimated Power Savings:       ~{power_saving*100:.1f}%")
    print(f"Estimated Area Savings:        ~{area_saving*100:.1f}%")
    print(f"Speed:                         Similar or faster")
    
    print(f"\n{'='*70}")
    print("QUALITY ASSESSMENT")
    print("="*70)
    
    # Quality classification based on MAE and WCE
    if mae < 1.0 and wce < 50:
        quality = "EXCELLENT (High accuracy, minimal errors)"
    elif mae < 2.0 and wce < 100:
        quality = "GOOD (Good accuracy, acceptable errors)"
    elif mae < 5.0:
        quality = "MEDIUM (Moderate accuracy)"
    else:
        quality = "POOR (Low accuracy, significant errors)"
    
    print(f"Overall Quality: {quality}")
    
    # Application suitability
    print(f"\nSuitable for:")
    if mae < 1.5:
        print("  ✓ Deep Neural Networks (CNNs, DNNs)")
        print("  ✓ Image Processing")
        print("  ✓ Signal Processing")
        print("  ✓ Machine Learning inference")
    elif mae < 5.0:
        print("  ✓ Deep Neural Networks (with careful tuning)")
        print("  ✓ Image Processing (non-critical applications)")
        print("  ~ Signal Processing (with verification)")
    else:
        print("  ✗ Not recommended for accuracy-critical applications")
    
    print(f"\n{'='*70}")
    print("EXAMPLE MULTIPLICATIONS")
    print("="*70)
    
    # Show some example calculations
    test_cases = [(10, 20), (50, 50), (100, 100), (200, 200), (255, 255)]
    print(f"{'A':<6} {'B':<6} {'Exact':<10} {'Approx':<10} {'Error':<10}")
    print("-"*70)
    for a, b in test_cases:
        exact = a * b
        approx = int(mul_table[a, b])
        error = approx - exact
        print(f"{a:<6} {b:<6} {exact:<10} {approx:<10} {error:<10}")
    
    print("="*70)

# Analyze mul8u_2P7
analyze_multiplier_bin('./multipliers/mul8u_2P7.bin')
