# utils.py
# Contains shared utility functions.

import numpy as np
from typing import Tuple

# Pre-calculated values for efficiency
SQRT2 = np.sqrt(2)

# Lookup table for mileage weighting factor omega
OMEGA_TABLE = {
    1: {0: 1.0, 45: 1.5, 90: 2.0},
    2: {0: SQRT2, 45: SQRT2 + 0.5, 90: SQRT2 + 1.0}
}

def get_omega(delta_L: int, delta_theta: int) -> float:
    """Looks up the mileage weighting factor omega from the table."""
    return OMEGA_TABLE.get(delta_L, {}).get(delta_theta, float('inf'))

def get_velocity_ms(slope_deg: float) -> float:
    """Returns vehicle velocity in m/s based on slope in degrees."""
    if 0 <= slope_deg < 10:
        speed_kmh = 30.0
    elif 10 <= slope_deg < 20:
        speed_kmh = 20.0
    elif 20 <= slope_deg <= 30: # Max slope is 30
        speed_kmh = 10.0
    else: # Should not be reachable on a valid path
        speed_kmh = 0
    return speed_kmh / 3.6

def coords_to_rc(x: int, y: int, map_dimension: int) -> Tuple[int, int]:
    """Converts problem coordinates (x, y) to numpy array indices (row, col)."""
    r = map_dimension - 1 - y
    c = x
    return int(r), int(c)