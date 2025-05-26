# 3D Photo - PLY to Splat Converter

A Python tool for converting binary PLY files containing Gaussian splat data to ASCII PLY format with `.splat` extension.

## Overview

This project provides a command-line utility to convert Gaussian splat data from binary PLY format to ASCII PLY format. Gaussian splatting is a 3D representation technique that stores point clouds with additional properties like spherical harmonics coefficients, opacity, scaling, and rotation parameters.

## Features

- Reads binary PLY files with Gaussian splat data
- Converts to ASCII PLY format with `.splat` extension
- Preserves all vertex properties including:
  - Position coordinates (x, y, z)
  - Normal vectors (nx, ny, nz)
  - Spherical harmonics coefficients (f_dc_*, f_rest_*)
  - Opacity values
  - Scale parameters (scale_0, scale_1, scale_2)
  - Rotation quaternions (rot_0, rot_1, rot_2, rot_3)
- Uses Typer for modern CLI interface
- Automatic output filename generation

## Installation

This project uses `uv` for dependency management. Install dependencies:

```bash
uv add typer numpy
```

## Usage

Basic usage:
```bash
uv run python ply_to_splat.py input.ply
```

This will create `input.splat` in the same directory.

Specify custom output file:
```bash
uv run python ply_to_splat.py input.ply -o custom_output.splat
```

View help:
```bash
uv run python ply_to_splat.py --help
```

## Example

Convert the sample JAPAN.ply file:
```bash
uv run python ply_to_splat.py assets/JAPAN.ply
```

This will create `assets/JAPAN.splat` containing the ASCII version of the Gaussian splat data.

## File Format

The output `.splat` files are ASCII PLY files with the structure:
```
ply
format ascii 1.0
element vertex [count]
property float x
property float y
property float z
...
end_header
[vertex data as space-separated float values]
```

## Requirements

- Python >= 3.10
- numpy >= 2.2.6
- typer >= 0.15.4