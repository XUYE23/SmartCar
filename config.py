# config.py
# This file contains all global constants and file paths.
# --- USER: PLEASE MODIFY THE FILE PATHS BELOW ---

# Input file paths
MAP_ELEVATION_PATH = "map.tif"
BAD_AREAS_PATH = "附件4：不良区域位置信息.xlsx" # e.g., "附件4：不良区域位置信息.xlsx"
PATH_P1_P2_PATH = "附件5：P1-P2的行驶路径.xlsx"     # e.g., "附件5：P1-P2行驶路径.xlsx"
PATH_P3_P4_PATH = "附件6：P3-P4的行驶路径.xlsx"     # e.g., "附件6：P3-P4行驶路径.xlsx"
PATH_P5_P6_PATH = "附件7：P5-P6行驶路径.xlsx"     # e.g., "附件7：P5-P6行驶路径.xlsx"

# Output file paths
PREPROCESSED_DATA_PATH = "preprocessed_data.npz"
PROBLEM1_PLOT_PATH = "problem1_results.png"
PROBLEM2_RESULTS_PATH = "problem2_results.xlsx"
PROBLEM3_RESULTS_PATH = "problem3_results.xlsx"

# --- DO NOT MODIFY BELOW THIS LINE ---

# Physical and simulation constants
CELLSIZE = 5.0  # meters
K_FACTOR = 5.0  # Elevation change factor from the problem description
MAP_DIMENSION = 12500

# Vehicle constants for 'A' type
MAX_SLOPE_DEG = 30.0
HEADINGS = [0, 45, 90, 135, 180, 225, 270, 315]