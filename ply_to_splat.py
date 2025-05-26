#!/usr/bin/env python3
import re
import struct
import numpy as np
import typer
from pathlib import Path
from typing import Optional


def parse_ply_header(file):
    """Parse PLY header and return property information."""
    header_lines = []
    properties = []

    while True:
        line = file.readline().decode("ascii").strip()
        header_lines.append(line)

        if line.startswith("property float"):
            prop_name = line.split()[-1]
            properties.append(prop_name)
        elif line == "end_header":
            break

    return properties, len(header_lines)


def read_ply_binary(filepath):
    """Read a binary PLY file containing Gaussian splat data."""
    with open(filepath, "rb") as file:
        # Parse header
        properties, header_lines = parse_ply_header(file)
        print(
            f"Parsed header with {len(properties)} properties: {', '.join(properties)}"
        )

        # Get number of vertices from header
        file.seek(0)
        vertex_count = None

        for line in file:
            line = line.decode("ascii").strip()
            if line == "end_header":
                break
            if match := re.match(r"element vertex (\d+)", line):
                vertex_count = int(match.group(1))

        if vertex_count is None:
            raise ValueError("Could not find vertex count in PLY header")

        print(f"Number of vertices: {vertex_count}")

        # Read binary data
        num_properties = len(properties)
        vertex_size = num_properties * 4  # 4 bytes per float

        vertices = []
        for _ in range(vertex_count):
            vertex_data = struct.unpack(f"<{num_properties}f", file.read(vertex_size))
            vertices.append(vertex_data)

        return np.array(vertices), properties


def write_splat_text(vertices, properties, output_path):
    """Write Gaussian splat data to text format."""
    with open(output_path, "w") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(vertices)}\n")

        for prop in properties:
            file.write(f"property float {prop}\n")
        file.write("end_header\n")

        # Write vertex data
        for vertex in vertices:
            formatted_values = [f"{val:.6f}" for val in vertex]
            file.write(" ".join(formatted_values) + "\n")


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
