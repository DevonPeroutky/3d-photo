# 3D Gaussian Splatting to Rhino Geometry

## Project Overview

Python3 script powering a Rhino 8 Grasshopper component that converts Gaussian Splatting PLY files into 3D geometry for CAD visualization and 3D printing workflows.

## Main Operations (`ply_to_rhino_script.py`)

### 1. PLY File Loading

- Reads Gaussian splat data from PLY files using `plyfile` library
- Transforms encoded data to real values:
  - **Scale**: log space � real space (`exp()` transformation)
  - **Opacity**: logit space � probability (sigmoid transformation)  
  - **Rotation**: quaternion normalization for valid rotations
- Extracts position, scale, rotation, color (spherical harmonics), and opacity

### 2. Data Processing Pipeline

- **Sampling**: Random sampling by percentage to reduce dataset size
- **Filtering**: Configurable noise reduction filters:
  - Opacity filtering (low transparency removal)
  - Brightness filtering (overly bright/dark removal)
  - Scale filtering (extreme size outlier removal)
  - Statistical outlier detection (k-nearest neighbor analysis)
  - Radius outlier detection (isolated point removal)
- **Normalization**: Centers point cloud around world origin
- **Coordinate Transformation**: Converts PLY space to Rhino coordinate system (X, Z, -Y)

### 3. Geometry Creation

**Render Modes**:

- `preview`: Fast mesh ellipsoids for real-time visualization
- `export`: High-quality Brep ellipsoids for CAD/3D printing
- `test`: Simple points for debugging and performance testing
- `merged`: Single mesh with vertex colors for high-performance 3D visualization (10K-50K splats)
- `pointcloud`: Single PointCloud object for ultra-fast rendering (30K+ splats)

**Transformation Pipeline**:

1. **Scale**: Creates ellipsoid from unit sphere using anisotropic scaling
2. **Rotate**: Applies quaternion rotation via manual matrix construction
3. **Coordinate Transform**: Applies PLY�Rhino coordinate mapping
4. **Translate**: Moves to final world position

**Performance Optimizations**:

- Mesh instancing with cached base sphere geometry
- Shared transformation pipeline for all geometry types

### 4. Color Processing

- Converts spherical harmonics DC coefficients to RGB colors
- Uses sigmoid activation function for color space transformation
- Returns paired geometry and color arrays for Grasshopper display
- Supports System.Drawing.Color format for Rhino integration

### 5. Development Features

- **Performance Monitoring**: Timing and memory usage tracking
- **Debug Mode**: Toggle between development and production configurations
- **Filter Configuration**: Centralized enable/disable system for all filters

## Key Dependencies

- **Rhino Integration**: `Rhino.Geometry`, `System.Drawing`, `rhinoscriptsyntax`
- **Data Processing**: `numpy`, `plyfile`, `psutil`
- **Custom Modules**: `performance_monitor`, `custom_types`, `color_utils`

## Usage in Grasshopper

Script expects these inputs from Grasshopper component:

- `file_path`: Path to PLY file
- `scale_factor`: Global scaling multiplier
- `subdivision_level`: Geometry detail level
- `sample_percentage`: Data sampling ratio (0.0-1.0)
- `render_mode`: Geometry type ("preview", "export", "test", "merged", "pointcloud")

Returns:

- `geometries`: List of Rhino geometry objects (Mesh/Brep/Point/PointCloud)
- `colors`: List of corresponding System.Drawing.Color objects

## Performance Guidelines

- **< 1K splats**: Any render mode works well
- **1K - 10K splats**: Use "preview" or "merged" for good performance
- **10K - 50K splats**: Use "merged" mode for mesh detail (~50-100x faster) or "pointcloud" for maximum speed (~100x faster)
- **> 50K splats**: Use "pointcloud" mode with aggressive filtering

