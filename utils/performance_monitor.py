"""
Performance monitoring utilities for tracking timing and memory usage.
"""

import time
import psutil


class PerformanceMonitor:
    """Monitor performance metrics including timing and memory usage."""

    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.operation = None

    def start_timing(self, operation_name: str):
        """Start timing an operation and record initial memory usage."""
        self.operation = operation_name
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        print(f"Starting {operation_name}...")

    def end_timing(self) -> dict:
        """End timing and return performance metrics."""
        if self.start_time is None:
            return {"duration": 0, "memory_used": 0, "memory_delta": 0}

        end_time = time.time()
        duration = end_time - self.start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_delta = end_memory - self.start_memory

        print(f"{self.operation} completed:")
        print(f"  Time: {duration:.2f} seconds")
        print(f"  Memory: {end_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")

        # Reset for next operation
        self.start_time = None
        self.start_memory = None
        self.operation = None

        return {
            "duration": duration,
            "memory_used": end_memory,
            "memory_delta": memory_delta,
        }

    def check_memory_usage(self):
        """Check current memory usage and warn if too high."""
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        if memory_mb > 1000:  # Warn at 1GB
            print(f"⚠️  High memory usage: {memory_mb:.1f} MB")
            print("Consider reducing splat count or mesh subdivision")
        elif memory_mb > 500:  # Info at 500MB
            print(f"📊 Current memory usage: {memory_mb:.1f} MB")

    def track_geometry_stats(self, geometries: list, render_mode: str):
        """Track and report geometry statistics."""
        if not geometries:
            return

        # Return geometry and colors for Grasshopper to display
        geometry_type = "Breps"
        if render_mode == "test":
            geometry_type = "Points"
        elif render_mode == "preview":
            geometry_type = "Meshes"
        else:
            geometry_type = "Breps"

        print(f"\n📊 Geometry Statistics ({render_mode} mode):")
        print(f"  Total {geometry_type}: {len(geometries):,}")

        # Count vertices/faces for meshes
        if geometry_type == "Meshes" and hasattr(geometries[0], "Vertices"):
            try:
                total_vertices = sum(
                    g.Vertices.Count for g in geometries if hasattr(g, "Vertices")
                )
                total_faces = sum(
                    g.Faces.Count for g in geometries if hasattr(g, "Faces")
                )
                avg_vertices = total_vertices / len(geometries) if geometries else 0
                avg_faces = total_faces / len(geometries) if geometries else 0

                print(f"  Total vertices: {total_vertices:,}")
                print(f"  Total faces: {total_faces:,}")
                print(f"  Avg vertices/mesh: {avg_vertices:.1f}")
                print(f"  Avg faces/mesh: {avg_faces:.1f}")
            except Exception as e:
                print(f"  Could not calculate mesh stats: {e}")

        # Estimate memory usage
        estimated_mb = len(geometries) * 0.1  # Rough estimate
        print(f"  Estimated geometry memory: ~{estimated_mb:.1f} MB")
