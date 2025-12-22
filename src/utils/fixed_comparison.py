#!/usr/bin/env python3
"""
Fixed GBR vs GBR+PINNS comparison
"""

import sys
import os
sys.path.append('src')

experiment = "localscalar_T16_to_qsq0"

print("=" * 70)
print("FIXED GBR vs GBR+PINNS COMPARISON")
print("=" * 70)
print(f"Dataset: {experiment}")
print()

# Run GBR first
print("STEP 1: Running GBR only...")
os.system(f'python -c "import sys; sys.path.append(\'src\'); import config; config.RUN_MODELS=[\'GBR\']; config.ENABLE_RATIO_METHOD=False; import lattice_qcd_analysis; sys.argv=[\'test\', \'{experiment}\']; lattice_qcd_analysis.main()"')

print("\n" + "="*50)
print("STEP 2: Running GBR+PINNS...")
os.system(f'python -c "import sys; sys.path.append(\'src\'); import config; config.RUN_MODELS=[\'GBR_PINNS\']; config.ENABLE_RATIO_METHOD=False; import lattice_qcd_analysis; sys.argv=[\'test\', \'{experiment}\']; lattice_qcd_analysis.main()"')

print("\n" + "=" * 70)
print("COMPARISON COMPLETE - Check outputs above!")
print("=" * 70)