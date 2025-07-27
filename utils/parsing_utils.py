import numpy as np
import typer
from pathlib import Path
from typing import Optional
from plyfile import PlyData, PlyElement


def read_ply_binary(filepath):
    """Read a binary PLY file containing Gaussian splat data."""
    plydata = PlyData.read(filepath)

    # Get vertex element
    vertex_element = plydata["vertex"]
    vertex_count = len(vertex_element)

    # Get property names
    properties = [prop.name for prop in vertex_element.properties]

    print(f"Parsed header with {len(properties)} properties: {', '.join(properties)}")
    print(f"Number of vertices: {vertex_count}")

    # Convert structured array to regular numpy array
    vertices = np.array([list(vertex) for vertex in vertex_element.data])

    return vertices, properties


def read_splat_text(filepath):
    """Read a text .splat file (which is actually PLY ASCII format)."""
    plydata = PlyData.read(filepath)
    
    # Get vertex element
    vertex_element = plydata["vertex"]
    vertex_count = len(vertex_element)
    
    # Get property names
    properties = [prop.name for prop in vertex_element.properties]
    
    print(f"Parsed header with {len(properties)} properties: {', '.join(properties)}")
    print(f"Number of vertices: {vertex_count}")
    
    # Convert structured array to regular numpy array
    vertices = np.array([list(vertex) for vertex in vertex_element.data])
    
    return vertices, properties


def write_ply_binary(vertices, properties, output_path):
    """Write Gaussian splat data to binary PLY format."""
    # Create structured array for plyfile
    dtype = [(prop, "f4") for prop in properties]
    vertex_data = np.array([tuple(vertex) for vertex in vertices], dtype=dtype)
    
    # Create PLY element
    vertex_element = PlyElement.describe(vertex_data, "vertex")
    
    # Write binary PLY file
    PlyData([vertex_element], text=False).write(output_path)


def write_splat_text(vertices, properties, output_path):
    """Write Gaussian splat data to text format."""
    # Create structured array for plyfile
    dtype = [(prop, "f4") for prop in properties]
    vertex_data = np.array([tuple(vertex) for vertex in vertices], dtype=dtype)

    # Create PLY element
    vertex_element = PlyElement.describe(vertex_data, "vertex")

    # Write ASCII PLY file
    PlyData([vertex_element], text=True).write(output_path)
