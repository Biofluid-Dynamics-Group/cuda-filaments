"""Convert icosa dat files to sphere seed files."""

import argparse
import math
import re
from pathlib import Path


def truncate_to_decimals(value, decimals=6):
    """Truncate a float to a specified number of decimal places."""
    multiplier = 10 ** decimals
    return math.trunc(value * multiplier) / multiplier


def parse_filename(filename):
    """Extract s and m parameters from filename icosa_d{s}_N{m}.dat."""
    match = re.match(r'icosa_d(\d+)_N(\d+)\.dat', filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match expected pattern")
    s = int(match.group(1))
    m = int(match.group(2))
    return s, m


def read_dat_file(filepath):
    """Read coordinates from dat file."""
    coordinates = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                coords = [float(x) for x in line.split()]
                if len(coords) != 3:
                    raise ValueError(f"Expected 3 coordinates per line, got {len(coords)}")
                coordinates.extend(coords)
    return coordinates


def write_seed_file(filepath, coordinates):
    """Write coordinates to seed file with 6 decimal places (truncated)."""
    truncated = [truncate_to_decimals(coord, 6) for coord in coordinates]
    formatted = [f"{coord:.6f}" for coord in truncated]
    with open(filepath, 'w') as f:
        f.write(' '.join(formatted) + '\n')


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(
        description='Convert icosa dat file to sphere seed file'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Input dat file (e.g., icosa_d5_N42.dat)'
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} not found")

    s, m = parse_filename(input_path.name)
    print(f"Parsed parameters: s={s}, m={m}")

    coordinates = read_dat_file(input_path)
    expected_count = m * 3
    if len(coordinates) != expected_count:
        raise ValueError(
            f"Expected {expected_count} coordinates ({m} rows × 3), "
            f"got {len(coordinates)}"
        )

    output_path = input_path.parent / f"sphere{m}.seed"
    write_seed_file(output_path, coordinates)
    print(f"Successfully wrote {output_path}")


if __name__ == '__main__':
    main()
