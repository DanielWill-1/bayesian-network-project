from cx_Freeze import setup, Executable
import sys
import os

# Include necessary packages and files
build_exe_options = {
    "packages": ["os", "sys", "pandas", "numpy", "matplotlib", "seaborn", "pgmpy", "torch", "sklearn", "xgboost"],
    "includes": [],
    "include_files": ["weatherAUS.csv"],  # Include your data file
    "excludes": [],
    "zip_include_packages": [],
}

# Base settings
base = None
if sys.platform == "win32":
    base = "Console"  # Use "Win32GUI" for GUI applications

# Target executable
executables = [
    Executable(
        "newBayesian.py",
        base=base,
        target_name="WeatherPredictionApp.exe",
    )
]

# Setup configuration
setup(
    name="WeatherPredictionApp",
    version="1.0",
    description="Weather Prediction Application",
    options={"build_exe": build_exe_options},
    executables=executables,
)
