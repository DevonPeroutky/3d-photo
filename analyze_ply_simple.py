#!/usr/bin/env python3
"""
Simple analysis of Gaussian splat PLY files to identify rendering issues.
"""

import math
import numpy as np


def print_data_ranges(sample, fields):
    """Print statistical analysis of raw data ranges."""
    print(f"\n{'=' * 70}")
    print("RAW DATA RANGES (First 1000 splats)")
    print("=" * 70)
    
    for category, field_list in fields.items():
        print(f"\n{category}:")
        for field in field_list:
            if field in sample.dtype.names:
                values = sample[field]
                print(
                    f"  {field:>8}: min={np.min(values):8.4f} max={np.max(values):8.4f} "
                    f"mean={np.mean(values):8.4f} std={np.std(values):8.4f}"
                )


def print_sample_transforms(verts, num_samples=5):
    """Print detailed transforms for sample splats."""
    print(f"\n{'=' * 70}")
    print(f"SAMPLE TRANSFORMS (First {num_samples} splats)")
    print("=" * 70)

    for i in range(num_samples):
        v = verts[i]
        print(f"\nSplat {i + 1}:")

        # Position
        pos = (float(v["x"]), float(v["y"]), float(v["z"]))
        print(f"  Position: {pos}")

        # Scale transform
        scale_raw = (float(v["scale_0"]), float(v["scale_1"]), float(v["scale_2"]))
        scale_exp = tuple(math.exp(s) for s in scale_raw)
        print(f"  Scale raw: {scale_raw}")
        print(f"  Scale exp: {scale_exp}")

        # Opacity transform
        opacity_raw = float(v["opacity"])
        opacity_sigmoid = 1.0 / (1.0 + math.exp(-opacity_raw))
        print(f"  Opacity raw: {opacity_raw:.4f} -> sigmoid: {opacity_sigmoid:.4f}")

        # Color transform
        color_raw = (float(v["f_dc_0"]), float(v["f_dc_1"]), float(v["f_dc_2"]))
        color_sigmoid = tuple(1.0 / (1.0 + math.exp(-c)) for c in color_raw)
        color_rgb = tuple(int(c * 255) for c in color_sigmoid)
        print(f"  Color raw: {color_raw}")
        print(f"  Color RGB: {color_rgb}")

        # Quaternion (both orderings)
        quat_raw = (
            float(v["rot_0"]),
            float(v["rot_1"]),
            float(v["rot_2"]),
            float(v["rot_3"]),
        )
        quat_norm = math.sqrt(sum(q * q for q in quat_raw))
        
        # Test as w,x,y,z
        quat_wxyz = quat_raw  # rot_0=w, rot_1=x, rot_2=y, rot_3=z
        
        # Test as x,y,z,w -> reorder to w,x,y,z
        quat_xyzw_reordered = (quat_raw[3], quat_raw[0], quat_raw[1], quat_raw[2])
        xyzw_norm = math.sqrt(sum(q * q for q in quat_xyzw_reordered))
        
        print(f"  Quat raw (rot_0,rot_1,rot_2,rot_3): {quat_raw} (norm: {quat_norm:.4f})")
        print(f"  If w,x,y,z: w={quat_wxyz[0]:.4f} x={quat_wxyz[1]:.4f} y={quat_wxyz[2]:.4f} z={quat_wxyz[3]:.4f}")
        print(f"  If x,y,z,w: w={quat_xyzw_reordered[0]:.4f} x={quat_xyzw_reordered[1]:.4f} y={quat_xyzw_reordered[2]:.4f} z={quat_xyzw_reordered[3]:.4f} (norm: {xyzw_norm:.4f})")


def detect_issues(sample):
    """Detect potential rendering issues in the data."""
    issues = []
    
    # Check scale distribution
    scale_fields = ["scale_0", "scale_1", "scale_2"]
    for field in scale_fields:
        values = sample[field]
        max_exp = math.exp(np.max(values))
        min_exp = math.exp(np.min(values))
        if max_exp > 100:
            issues.append(
                f"⚠️  {field}: Very large ellipsoids after exp() (max: {max_exp:.2f})"
            )
        if min_exp < 0.0001:
            issues.append(
                f"⚠️  {field}: Very small ellipsoids after exp() (min: {min_exp:.6f})"
            )

    # Check position spread
    pos_fields = ["x", "y", "z"]
    for field in pos_fields:
        values = sample[field]
        range_val = np.max(values) - np.min(values)
        if range_val > 1000:
            issues.append(
                f"⚠️  {field}: Very large coordinate range ({range_val:.2f})"
            )
            
    return issues


def analyze_quaternion_ordering(sample):
    """Analyze quaternion data to determine likely ordering (w,x,y,z vs x,y,z,w)."""
    quat_issues = 0
    w_first_normalized = 0
    x_first_normalized = 0
    
    for i in range(min(100, len(sample))):
        v = sample[i]
        quat = [float(v[f]) for f in ["rot_0", "rot_1", "rot_2", "rot_3"]]
        norm = math.sqrt(sum(q * q for q in quat))
        
        if abs(norm - 1.0) > 0.1:
            quat_issues += 1
        
        # Test both quaternion orderings to see which produces better normalized quaternions
        if norm > 0:
            # Assume w,x,y,z ordering
            w_quat = [quat[0], quat[1], quat[2], quat[3]]  # w,x,y,z
            w_norm = math.sqrt(sum(q * q for q in w_quat))
            if abs(w_norm - 1.0) < 0.05:  # Well normalized
                w_first_normalized += 1
            
            # Assume x,y,z,w ordering  
            x_quat = [quat[3], quat[0], quat[1], quat[2]]  # x,y,z,w -> w,x,y,z
            x_norm = math.sqrt(sum(q * q for q in x_quat))
            if abs(x_norm - 1.0) < 0.05:  # Well normalized
                x_first_normalized += 1
    
    return quat_issues, w_first_normalized, x_first_normalized


def print_quaternion_analysis(w_first_normalized, x_first_normalized):
    """Print quaternion ordering analysis results."""
    print(f"\n{'=' * 70}")
    print("QUATERNION ORDERING ANALYSIS")
    print("=" * 70)
    
    if w_first_normalized > x_first_normalized:
        print(f"✅ LIKELY w,x,y,z ordering: {w_first_normalized}/100 samples well-normalized")
        print(f"   x,y,z,w ordering: {x_first_normalized}/100 samples well-normalized")
        print("   📝 Quaternion fields appear to be: rot_0=w, rot_1=x, rot_2=y, rot_3=z")
    elif x_first_normalized > w_first_normalized:
        print(f"✅ LIKELY x,y,z,w ordering: {x_first_normalized}/100 samples well-normalized")
        print(f"   w,x,y,z ordering: {w_first_normalized}/100 samples well-normalized")
        print("   📝 Quaternion fields appear to be: rot_0=x, rot_1=y, rot_2=z, rot_3=w")
    else:
        print(f"❓ UNCLEAR ordering: w,x,y,z={w_first_normalized}, x,y,z,w={x_first_normalized}")
        print("   📝 May need manual inspection or both orderings produce similar results")


def print_recommendations():
    """Print rendering issue causes and recommended fixes."""
    print(f"\n{'=' * 70}")
    print("LIKELY CAUSES OF JUMBLED RENDERING:")
    print("=" * 70)
    print("1. 🔴 SCALE: Raw scale values are in log space - need exp() transform")
    print("2. 🔴 OPACITY: Raw opacity in logit space - need sigmoid() transform")
    print("3. 🔴 COLORS: SH coefficients need sigmoid() activation")
    print("4. 🟡 QUATERNIONS: May need normalization and correct ordering")
    print("5. 🟡 COORDINATE SYSTEM: Check if positions need adjustment")

    print(f"\n{'=' * 70}")
    print("RECOMMENDED FIXES:")
    print("=" * 70)
    print("✅ Apply exp(scale_i) before creating ellipsoid geometry")
    print("✅ Apply sigmoid(opacity) for transparency")
    print("✅ Apply sigmoid(f_dc_i) then scale to RGB [0,255]")
    print("✅ Normalize quaternions before applying rotation")


def analyze_ply_data(file_path: str):
    """Analyze PLY file and show what's causing the rendering issues."""

    try:
        import plyfile

        ply = plyfile.PlyData.read(file_path)
        verts = ply["vertex"].data

        print(f"Analyzing: {file_path}")
        print(f"Total splats: {len(verts)}")
        print(f"Fields: {list(verts.dtype.names)}")

        # Prepare sample data
        sample_size = min(1000, len(verts))
        sample = verts[:sample_size]

        # Define field categories
        fields = {
            "Position": ["x", "y", "z"],
            "Scale": ["scale_0", "scale_1", "scale_2"],
            "Rotation": ["rot_0", "rot_1", "rot_2", "rot_3"],
            "Color (SH)": ["f_dc_0", "f_dc_1", "f_dc_2"],
            "Opacity (logit)": ["opacity"],
        }

        # Run modular analysis functions
        print_data_ranges(sample, fields)
        print_sample_transforms(verts, num_samples=5)
        
        # Issue detection
        print(f"\n{'=' * 70}")
        print("ISSUE ANALYSIS")
        print("=" * 70)
        
        issues = detect_issues(sample)
        quat_issues, w_first_normalized, x_first_normalized = analyze_quaternion_ordering(sample)
        
        if quat_issues > 10:
            issues.append(f"⚠️  Quaternions: {quat_issues}/100 samples not normalized")

        if issues:
            print("DETECTED ISSUES:")
            for issue in issues:
                print(issue)
        else:
            print("✅ No major issues detected")

        # Print analysis results and recommendations
        print_quaternion_analysis(w_first_normalized, x_first_normalized)
        print_recommendations()

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

