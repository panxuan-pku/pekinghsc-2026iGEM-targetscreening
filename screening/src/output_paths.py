"""Preflight file destinations for the core command-line stages."""
from pathlib import Path


def validate_output_paths(inputs, outputs):
    inputs = [Path(p).resolve() for p in inputs if p is not None]
    outputs = [Path(p).resolve() for p in outputs]
    checked = []
    for output in outputs:
        if output.exists() and not output.is_file():
            raise ValueError(f"output path conflict: {output} is not a file")
        for parent in output.parents:
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"output path conflict: parent {parent} is not a directory")
        for other in inputs + checked:
            if (output == other or (output.exists() and other.exists() and output.samefile(other))
                    or output in other.parents or other in output.parents):
                raise ValueError(f"output path conflict: {output} overlaps {other}")
        checked.append(output)
