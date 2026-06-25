import numpy as np
import os
import re


_FLOAT_RE = re.compile(
    r'(?<![A-Za-z0-9])([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)'
)


def read_mako_points(filepath):
    """
    Parse a MakoPoints.txt file.
    Returns a dict of landmark_name -> np.array([X, Y, Z]).

    Skips header/legend lines by checking line content rather than
    line number — more robust than MATLAB's hardcoded i=13:19 loop.

    Coordinate columns may be tab- or space-separated; some exports merge
    adjacent coords into one field (e.g. S030), so we extract the first
    three floats on each line instead of relying on tab splits alone.
    """
    SKIP_TOKENS = {'Legend', 'Point:', 'Name', 'Data', '======', '====', ''}

    landmarks = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            nums = _FLOAT_RE.findall(line)
            if len(nums) < 3:
                continue

            first_num = _FLOAT_RE.search(line)
            name = line[:first_num.start()].strip()
            if not name or name in SKIP_TOKENS:
                continue

            try:
                coords = np.array([float(nums[0]),
                                   float(nums[1]),
                                   float(nums[2])])
            except ValueError:
                continue

            landmarks[name] = coords

    return landmarks


