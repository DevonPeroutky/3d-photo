#!/usr/bin/env python3
"""
Debug script to analyze Gaussian splat data and identify rendering issues.
Uses the existing GaussianSplatReader to examine actual parsed data.
"""

import sys
import os
import math
import numpy as np

# Add utils to path so we can import modules
sys.path.append('/Users/devonperoutky/Development/projects/3d-photo/utils/')

# Import our existing classes
from gaussian_splat import GaussianSplat


def create_debug_reader():
    """Create a debug version of GaussianSplatReader with analysis capabilities."""
    
    class DebugGaussianSplatReader:
        def load_gaussian_splats(self, file_path: str):
            """Load and analyze PLY file data."""
            try:
                import plyfile
                ply = plyfile.PlyData.read(file_path)
                verts = ply["vertex"].data
                
                print(f"Loaded PLY file: {file_path}")
                print(f"Total splats: {len(verts)}")
                print(f"Available fields: {list(verts.dtype.names)}")
                
                # Analyze raw data ranges
                self.analyze_raw_data(verts)
                
                # Convert to simplified data for analysis
                splats_raw = []
                for i, v in enumerate(verts[:10]):  # Just first 10 for analysis
                    # Create simple data structure for analysis
                    splat_data = {
                        'position': (float(v["x"]), float(v["y"]), float(v["z"])),
                        'scale': (float(v["scale_0"]), float(v["scale_1"]), float(v["scale_2"])),
                        'rotation_angles': (float(v["rot_0"]), float(v["rot_1"]), float(v["rot_2"]), float(v["rot_3"])),
                        'color': (float(v["f_dc_0"]), float(v["f_dc_1"]), float(v["f_dc_2"])),
                        'opacity': float(v["opacity"]),
                        'normal': (float(v["nx"]), float(v["ny"]), float(v["nz"]))
                    }
                    splats_raw.append(splat_data)
                
                # Show before/after transforms
                self.show_transform_comparison(splats_raw)
                
                return splats_raw
                
            except ImportError:
                print("plyfile not available, creating mock data for analysis")
                return []
        
        def analyze_raw_data(self, verts):
            """Analyze statistical properties of raw PLY data."""
            
            print(f"\n{'='*60}")
            print("RAW DATA ANALYSIS")
            print("="*60)
            
            fields_to_analyze = {
                'Position': ['x', 'y', 'z'],
                'Scale (Log)': ['scale_0', 'scale_1', 'scale_2'], 
                'Rotation': ['rot_0', 'rot_1', 'rot_2', 'rot_3'],
                'Color (SH)': ['f_dc_0', 'f_dc_1', 'f_dc_2'],
                'Opacity (Logit)': ['opacity'],
                'Normals': ['nx', 'ny', 'nz']
            }
            
            for category, fields in fields_to_analyze.items():
                print(f"\n{category}:")
                for field in fields:
                    if field in verts.dtype.names:
                        values = verts[field]
                        print(f"  {field:>8}: min={np.min(values):8.4f}, max={np.max(values):8.4f}, "
                              f"mean={np.mean(values):8.4f}, std={np.std(values):8.4f}")
        
        def show_transform_comparison(self, splats_raw):
            """Show before/after mathematical transforms."""
            
            print(f"\n{'='*60}")
            print("TRANSFORM COMPARISON (First 5 splats)")
            print("="*60)
            
            for i, splat in enumerate(splats_raw[:5]):
                print(f"\nSplat {i+1}:")
                
                # Position (no transform)
                print(f"  Position: {splat['position']}")
                
                # Scale: exp(raw) transform
                scale_raw = splat['scale']
                scale_exp = tuple(math.exp(s) for s in scale_raw)
                print(f"  Scale raw:    {scale_raw}")
                print(f"  Scale exp():  {scale_exp}")
                
                # Opacity: sigmoid transform  
                opacity_raw = splat['opacity']
                opacity_sigmoid = 1.0 / (1.0 + math.exp(-opacity_raw))
                print(f"  Opacity raw:  {opacity_raw:.4f}")
                print(f"  Opacity sig:  {opacity_sigmoid:.4f}")
                
                # Color: sigmoid transform
                color_raw = splat['color']
                color_sigmoid = tuple(1.0 / (1.0 + math.exp(-c)) for c in color_raw)
                color_rgb = tuple(int(c * 255) for c in color_sigmoid)
                print(f"  Color raw:    {color_raw}")
                print(f"  Color RGB:    {color_rgb}")
                
                # Quaternion: normalize
                quat_raw = splat['rotation_angles']
                quat_norm = math.sqrt(sum(q*q for q in quat_raw))
                quat_normalized = tuple(q/quat_norm for q in quat_raw) if quat_norm > 0 else quat_raw
                print(f"  Quat raw:     {quat_raw} (norm: {quat_norm:.4f})")
                print(f"  Quat norm:    {quat_normalized}")
        
        def identify_issues(self, splats):
            """Identify potential rendering issues."""
            
            print(f"\n{'='*60}")
            print("IDENTIFIED ISSUES")
            print("="*60)
            
            issues = []
            
            for i, splat in enumerate(splats[:10]):
                # Check for extreme scale values
                scale_exp = tuple(math.exp(s) for s in splat['scale'])
                max_scale = max(scale_exp)
                min_scale = min(scale_exp)
                
                if max_scale > 50:
                    issues.append(f"Splat {i}: Very large scale {max_scale:.2f}")
                if min_scale < 0.001:
                    issues.append(f"Splat {i}: Very small scale {min_scale:.6f}")
                
                # Check quaternion
                quat_norm = math.sqrt(sum(q*q for q in splat['rotation_angles']))
                if abs(quat_norm - 1.0) > 0.2:
                    issues.append(f"Splat {i}: Unnormalized quaternion (norm: {quat_norm:.4f})")
            
            if issues:
                for issue in issues:
                    print(f"⚠️  {issue}")
            else:
                print("✅ No major issues detected in sample data")
    
    return DebugGaussianSplatReader()


def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_splat_data.py <path_to_ply_file>")
        print("Example: python debug_splat_data.py ../assets/JAPAN.ply")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    print(f"Debugging Gaussian splat file: {file_path}")
    
    reader = create_debug_reader()
    splats = reader.load_gaussian_splats(file_path)
    
    if splats:
        reader.identify_issues(splats)
        
        print(f"\n{'='*60}")
        print("RECOMMENDED FIXES:")
        print("="*60)
        print("1. Apply exp() to scale parameters before creating ellipsoids")
        print("2. Apply sigmoid() to opacity values")
        print("3. Apply sigmoid() to color SH coefficients")
        print("4. Normalize quaternion rotation parameters")
        print("5. Check if position coordinates need scaling/translation")


if __name__ == "__main__":
    main()