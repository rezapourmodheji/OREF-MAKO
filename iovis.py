import numpy as np
import os
import re


def read_mako_points(filepath):
    """
    Parse a MakoPoints.txt file.
    Returns a dict of landmark_name -> np.array([X, Y, Z]).

    Skips header/legend lines by checking line content rather than
    line number — more robust than MATLAB's hardcoded i=13:19 loop.
    """
    SKIP_TOKENS = {'Legend', 'Point:', 'Name', 'Data', '======', '====', ''}

    landmarks = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            parts = [p.strip() for p in parts if p.strip() != '']

            # Need at least: name + X + Y + Z
            if len(parts) < 4:
                continue
            if parts[0] in SKIP_TOKENS:
                continue

            name = parts[0]
            try:
                coords = np.array([float(parts[1]),
                                   float(parts[2]),
                                   float(parts[3])])
            except ValueError:
                continue  # skip any line that doesn't parse cleanly

            landmarks[name] = coords

    return landmarks


