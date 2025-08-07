# Gaussian Splat to Rhino Geometry Transformation Pipeline

## Data Loading & Initial Transformations

1. **PLY File Loading**
   - Scale values: `log space → real space` using `exp()`
   - Quaternions: Normalize to unit length `[w,x,y,z]`
   - Opacity: `logit space → probability` using sigmoid `1/(1+exp(-x))`

2. **Preprocessing**
   - Normalize positions around world origin
   - Regional sampling within radius of centroid
   - Apply filters (opacity, brightness, scale, outliers)

## Individual Splat Transformation Chain (4 Steps)

For each Gaussian splat → ellipsoid mesh:

### Step 1: Scale Transform
```
geometry.Transform(Scale(WorldXY, scale_x, scale_y, scale_z))
```
- Unit sphere → anisotropic ellipsoid
- Scale = `splat.scale * scale_multiplier * 2.5`
- Converts Gaussian σ → visual extent (2-3σ = 95-99% distribution)

### Step 2: Quaternion Rotation
```
geometry.Transform(quaternion_to_rotation_transform(splat.rotation_angles))
```
- Apply splat's 3D orientation using normalized quaternion
- Uses Rhino's `Quaternion.GetRotation()` API

### Step 3: Coordinate System Transform
```
Manual matrix transformation:
[1  0  0]   PLY: (X, Y, Z) → Rhino: (X, Z, -Y)
[0  0  1]   
[0 -1  0]   
```
- X axis: unchanged (`X → X`)
- Y axis: becomes Z (`Y → Z`) 
- Z axis: becomes negative Y (`Z → -Y`)

### Step 4: Translation
```
transformed_pos = Point3d(splat.position.X, splat.position.Z, -splat.position.Y)
geometry.Transform(Translation(transformed_pos))
```
- Move to final world position with coordinate transform applied

## Color Processing

```
r = sh_to_rgb(splat.color[0])  # f_dc_0 → Red
g = sh_to_rgb(splat.color[1])  # f_dc_1 → Green  
b = sh_to_rgb(splat.color[2])  # f_dc_2 → Blue
alpha = splat.opacity * 255    # Opacity → Alpha
```
- Spherical harmonics DC coefficients → RGB using sigmoid
- Opacity converted to alpha channel

## Coordinate System Mapping

| PLY/NeRF Space | → | Rhino Space |
|----------------|---|-------------|
| X (right)      | → | X (right)   |
| Y (up)         | → | Z (up)      |
| Z (forward)    | → | -Y (back)   |

## Scale Interpretation Chain

1. **Raw PLY**: Standard deviation (σ) in log space
2. **Real space**: `exp(log_scale)` → actual σ values  
3. **Visual extent**: `σ * scale_multiplier * 2.5` → ellipsoid size
4. **Purpose**: Show 95% of Gaussian distribution (≈2.5σ)