#!/usr/bin/env python3
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


def write_splat_text(vertices, properties, output_path):
    """Write Gaussian splat data to text format."""
    # Create structured array for plyfile
    dtype = [(prop, "f4") for prop in properties]
    vertex_data = np.array([tuple(vertex) for vertex in vertices], dtype=dtype)

    # Create PLY element
    vertex_element = PlyElement.describe(vertex_data, "vertex")

    # Write ASCII PLY file
    PlyData([vertex_element], text=True).write(output_path)


def main(
    input_file: Path = typer.Argument(..., help="Input PLY file path"),
    output_file: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output text file path (default: input_name.splat)"
    ),
):
    """Convert PLY Gaussian splat binary files to text format."""
    if not input_file.exists():
        typer.echo(f"Error: Input file {input_file} does not exist", err=True)
        raise typer.Exit(1)

    # Determine output path - always use .splat extension
    if output_file:
        output_path = output_file.with_suffix(".splat")
    else:
        output_path = input_file.with_suffix(".splat")

    try:
        typer.echo(f"Reading PLY file: {input_file}")
        vertices, properties = read_ply_binary(input_file)

        typer.echo(f"Found {len(vertices)} vertices with {len(properties)} properties")
        typer.echo(
            f"Properties: {', '.join(properties[:10])}{'...' if len(properties) > 10 else ''}"
        )

        typer.echo(f"Writing text splat file: {output_path}")
        write_splat_text(vertices, properties, output_path)

        typer.echo("Conversion completed successfully!")

    except Exception as e:
        typer.echo(f"Error during conversion: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
