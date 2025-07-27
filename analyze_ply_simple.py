#!/usr/bin/env python3
"""
Simple analysis of Gaussian splat PLY files to identify rendering issues.
"""

import math
import numpy as np


def analyze_ply_data(file_path: str):
    """Analyze PLY file and show what's causing the rendering issues."""
    
    try:
        import plyfile
        ply = plyfile.PlyData.read(file_path)
        verts = ply["vertex"].data
        
        print(f"Analyzing: {file_path}")
        print(f"Total splats: {len(verts)}")
        print(f"Fields: {list(verts.dtype.names)}")
        
        # Statistical analysis
        print(f"\n{'='*70}")
        print("RAW DATA RANGES (First 1000 splats)")
        print("="*70)
        
        sample_size = min(1000, len(verts))
        sample = verts[:sample_size]
        
        fields = {
            'Position': ['x', 'y', 'z'],
            'Scale (log)': ['scale_0', 'scale_1', 'scale_2'],
            'Rotation': ['rot_0', 'rot_1', 'rot_2', 'rot_3'], 
            'Color (SH)': ['f_dc_0', 'f_dc_1', 'f_dc_2'],
            'Opacity (logit)': ['opacity']
        }
        
        for category, field_list in fields.items():
            print(f"\n{category}:")
            for field in field_list:
                if field in sample.dtype.names:
                    values = sample[field]
                    print(f"  {field:>8}: min={np.min(values):8.4f} max={np.max(values):8.4f} "
                          f"mean={np.mean(values):8.4f} std={np.std(values):8.4f}")
        
        # Show sample transforms
        print(f"\n{'='*70}")
        print("SAMPLE TRANSFORMS (First 5 splats)")
        print("="*70)
        
        for i in range(min(5, len(verts))):
            v = verts[i]
            print(f"\nSplat {i+1}:")
            
            # Position
            pos = (float(v['x']), float(v['y']), float(v['z']))
            print(f"  Position: {pos}")
            
            # Scale transform
            scale_raw = (float(v['scale_0']), float(v['scale_1']), float(v['scale_2']))
            scale_exp = tuple(math.exp(s) for s in scale_raw)
            print(f"  Scale raw: {scale_raw}")
            print(f"  Scale exp: {scale_exp}")
            
            # Opacity transform
            opacity_raw = float(v['opacity'])
            opacity_sigmoid = 1.0 / (1.0 + math.exp(-opacity_raw))
            print(f"  Opacity raw: {opacity_raw:.4f} -> sigmoid: {opacity_sigmoid:.4f}")
            
            # Color transform
            color_raw = (float(v['f_dc_0']), float(v['f_dc_1']), float(v['f_dc_2']))
            color_sigmoid = tuple(1.0 / (1.0 + math.exp(-c)) for c in color_raw)
            color_rgb = tuple(int(c * 255) for c in color_sigmoid)
            print(f"  Color raw: {color_raw}")
            print(f"  Color RGB: {color_rgb}")
            
            # Quaternion
            quat = (float(v['rot_0']), float(v['rot_1']), float(v['rot_2']), float(v['rot_3']))
            quat_norm = math.sqrt(sum(q*q for q in quat))
            quat_normalized = tuple(q/quat_norm for q in quat) if quat_norm > 0 else quat
            print(f"  Quat raw: {quat} (norm: {quat_norm:.4f})")
            print(f"  Quat normalized: {quat_normalized}")
        
        # Issue identification
        print(f"\n{'='*70}")
        print("ISSUE ANALYSIS")
        print("="*70)
        
        issues = []
        
        # Check scale distribution
        scale_fields = ['scale_0', 'scale_1', 'scale_2']
        for field in scale_fields:
            values = sample[field]
            max_exp = math.exp(np.max(values))
            min_exp = math.exp(np.min(values))
            if max_exp > 100:
                issues.append(f"⚠️  {field}: Very large ellipsoids after exp() (max: {max_exp:.2f})")
            if min_exp < 0.0001:
                issues.append(f"⚠️  {field}: Very small ellipsoids after exp() (min: {min_exp:.6f})")
        
        # Check position spread
        pos_fields = ['x', 'y', 'z']
        for field in pos_fields:
            values = sample[field]
            range_val = np.max(values) - np.min(values)
            if range_val > 1000:
                issues.append(f"⚠️  {field}: Very large coordinate range ({range_val:.2f})")
        
        # Check quaternion normalization
        quat_issues = 0
        for i in range(min(100, len(sample))):
            v = sample[i]
            quat = [float(v[f]) for f in ['rot_0', 'rot_1', 'rot_2', 'rot_3']]
            norm = math.sqrt(sum(q*q for q in quat))
            if abs(norm - 1.0) > 0.1:
                quat_issues += 1
        
        if quat_issues > 10:
            issues.append(f"⚠️  Quaternions: {quat_issues}/100 samples not normalized")
        
        if issues:
            print("DETECTED ISSUES:")
            for issue in issues:
                print(issue)
        else:
            print("✅ No major issues detected")
        
        print(f"\n{'='*70}")
        print("LIKELY CAUSES OF JUMBLED RENDERING:")
        print("="*70)
        print("1. 🔴 SCALE: Raw scale values are in log space - need exp() transform")
        print("2. 🔴 OPACITY: Raw opacity in logit space - need sigmoid() transform") 
        print("3. 🔴 COLORS: SH coefficients need sigmoid() activation")
        print("4. 🟡 QUATERNIONS: May need normalization")
        print("5. 🟡 COORDINATE SYSTEM: Check if positions need adjustment")
        
        print(f"\n{'='*70}")
        print("RECOMMENDED FIXES:")
        print("="*70)
        print("✅ Apply exp(scale_i) before creating ellipsoid geometry")
        print("✅ Apply sigmoid(opacity) for transparency")
        print("✅ Apply sigmoid(f_dc_i) then scale to RGB [0,255]")
        print("✅ Normalize quaternions before applying rotation")
        
    except ImportError:
        print("ERROR: plyfile module not found. Install with: pip install plyfile")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_ply_simple.py <path_to_ply_file>")
        sys.exit(1)
    
    analyze_ply_data(sys.argv[1])