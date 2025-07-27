#!/usr/bin/env python3
"""
Tool to analyze Gaussian splat PLY files and identify data issues.
This helps debug why the rendered ellipsoids appear jumbled.
"""

import numpy as np
import plyfile
import math
from typing import Dict, Any
import sys


def analyze_ply_data(file_path: str) -> Dict[str, Any]:
    """Analyze PLY file and return statistical summary of all fields."""
    
    print(f"Analyzing PLY file: {file_path}")
    ply = plyfile.PlyData.read(file_path)
    verts = ply["vertex"].data
    
    print(f"Total vertices: {len(verts)}")
    print(f"Available fields: {verts.dtype.names}")
    
    analysis = {}
    
    # Analyze each field
    for field in verts.dtype.names:
        values = verts[field]
        analysis[field] = {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
        }
    
    return analysis, verts


def print_analysis(analysis: Dict[str, Any]):
    """Print formatted analysis results."""
    
    print("\n" + "="*80)
    print("GAUSSIAN SPLAT DATA ANALYSIS")
    print("="*80)
    
    # Group fields by category
    position_fields = ['x', 'y', 'z']
    normal_fields = ['nx', 'ny', 'nz'] 
    color_fields = ['f_dc_0', 'f_dc_1', 'f_dc_2']
    scale_fields = ['scale_0', 'scale_1', 'scale_2']
    rotation_fields = ['rot_0', 'rot_1', 'rot_2', 'rot_3']
    
    categories = [
        ("POSITION", position_fields),
        ("NORMALS", normal_fields),
        ("COLORS (Spherical Harmonics DC)", color_fields),
        ("SCALE (Log Space)", scale_fields),
        ("ROTATION (Quaternion)", rotation_fields),
        ("OPACITY (Logit Space)", ['opacity'])
    ]
    
    for category_name, fields in categories:
        print(f"\n{category_name}:")
        print("-" * 40)
        
        for field in fields:
            if field in analysis:
                stats = analysis[field]
                print(f"{field:>8}: min={stats['min']:8.4f}, max={stats['max']:8.4f}, "
                      f"mean={stats['mean']:8.4f}, std={stats['std']:8.4f}")


def identify_issues(analysis: Dict[str, Any], verts) -> list:
    """Identify potential issues with the data."""
    
    issues = []
    
    # Check scale values - should be in log space, so negative values are normal
    scale_fields = ['scale_0', 'scale_1', 'scale_2']
    for field in scale_fields:
        if field in analysis:
            stats = analysis[field]
            # After exp transform, very large scales could cause issues
            max_exp_scale = math.exp(stats['max'])
            if max_exp_scale > 100:
                issues.append(f"Scale {field}: exp({stats['max']:.2f}) = {max_exp_scale:.2f} - very large ellipsoids")
            if stats['min'] < -10:
                issues.append(f"Scale {field}: min {stats['min']:.2f} -> exp = {math.exp(stats['min']):.6f} - very small ellipsoids")
    
    # Check opacity values - should be in logit space
    if 'opacity' in analysis:
        stats = analysis['opacity']
        # Convert some samples to see sigmoid results
        sigmoid_min = 1.0 / (1.0 + math.exp(-stats['min']))
        sigmoid_max = 1.0 / (1.0 + math.exp(-stats['max']))
        issues.append(f"Opacity range: {stats['min']:.2f} to {stats['max']:.2f}")
        issues.append(f"After sigmoid: {sigmoid_min:.4f} to {sigmoid_max:.4f}")
    
    # Check quaternion normalization
    rotation_fields = ['rot_0', 'rot_1', 'rot_2', 'rot_3']
    if all(field in analysis for field in rotation_fields):
        # Sample a few quaternions to check if they're normalized
        sample_indices = np.random.choice(len(verts), min(100, len(verts)), replace=False)
        unnormalized_count = 0
        
        for i in sample_indices:
            quat = np.array([verts[field][i] for field in rotation_fields])
            norm = np.linalg.norm(quat)
            if abs(norm - 1.0) > 0.1:  # Allow some tolerance
                unnormalized_count += 1
        
        if unnormalized_count > 0:
            issues.append(f"Quaternions: {unnormalized_count}/{len(sample_indices)} samples are not normalized")
    
    # Check position spread
    position_fields = ['x', 'y', 'z']
    for field in position_fields:
        if field in analysis:
            stats = analysis[field]
            range_val = stats['max'] - stats['min']
            if range_val > 1000:
                issues.append(f"Position {field}: very large range {range_val:.2f}")
    
    return issues


def show_sample_transforms(verts, num_samples=5):
    """Show before/after transforms for sample data points."""
    
    print(f"\n{'='*80}")
    print("SAMPLE TRANSFORMS (Before -> After)")
    print("="*80)
    
    indices = np.random.choice(len(verts), min(num_samples, len(verts)), replace=False)
    
    for i, idx in enumerate(indices):
        print(f"\nSample {i+1} (index {idx}):")
        
        # Position (no transform needed)
        pos = (verts['x'][idx], verts['y'][idx], verts['z'][idx])
        print(f"  Position: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        
        # Scale transform: exp(scale_i)
        scale_raw = (verts['scale_0'][idx], verts['scale_1'][idx], verts['scale_2'][idx])
        scale_exp = tuple(math.exp(s) for s in scale_raw)
        print(f"  Scale:    {scale_raw} -> {scale_exp}")
        
        # Opacity transform: sigmoid(opacity)
        opacity_raw = verts['opacity'][idx]
        opacity_sigmoid = 1.0 / (1.0 + math.exp(-opacity_raw))
        print(f"  Opacity:  {opacity_raw:.4f} -> {opacity_sigmoid:.4f}")
        
        # Quaternion (check normalization)
        quat_raw = (verts['rot_0'][idx], verts['rot_1'][idx], 
                   verts['rot_2'][idx], verts['rot_3'][idx])
        quat_norm = np.linalg.norm(quat_raw)
        quat_normalized = tuple(q/quat_norm for q in quat_raw)
        print(f"  Quaternion: {quat_raw} (norm: {quat_norm:.4f})")
        print(f"  Normalized: {quat_normalized}")
        
        # Color transform: sigmoid(f_dc_i)
        color_raw = (verts['f_dc_0'][idx], verts['f_dc_1'][idx], verts['f_dc_2'][idx])
        color_sigmoid = tuple(1.0 / (1.0 + math.exp(-c)) for c in color_raw)
        color_rgb = tuple(int(c * 255) for c in color_sigmoid)
        print(f"  Color SH:  {color_raw}")
        print(f"  Color RGB: {color_rgb}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_ply.py <path_to_ply_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        analysis, verts = analyze_ply_data(file_path)
        print_analysis(analysis)
        
        issues = identify_issues(analysis, verts)
        
        if issues:
            print(f"\n{'='*80}")
            print("IDENTIFIED ISSUES:")
            print("="*80)
            for issue in issues:
                print(f"⚠️  {issue}")
        
        show_sample_transforms(verts)
        
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS:")
        print("="*80)
        print("1. Apply exp() transform to scale_0, scale_1, scale_2")
        print("2. Apply sigmoid() transform to opacity")  
        print("3. Apply sigmoid() transform to f_dc_0, f_dc_1, f_dc_2 for colors")
        print("4. Normalize quaternions (rot_0, rot_1, rot_2, rot_3)")
        print("5. Consider scaling down very large position ranges if needed")
        
    except Exception as e:
        print(f"Error analyzing PLY file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()