import os

# List of system dependencies
system_packages = [
    "libdrm-dev",
    "libcamera-dev",
    "libcap-dev",
    "ffmpeg",
    "libavcodec-dev",
    "libavdevice-dev",
    "libavfilter-dev",
    "libavformat-dev",
    "libavutil-dev",
    "libswscale-dev",
    "libswresample-dev",
    "libpostproc-dev"
]

# Install system dependencies (requires sudo)
print("Installing system dependencies...")
os.system(f"sudo apt update && sudo apt install -y {' '.join(system_packages)}")

# List of Python packages
python_packages = [
    "picamera2",
    "numpy",
    "pillow",
    "simplejpeg",
    "v4l2-python3",
    "python-prctl",
    "piexif"
]

# Install Python packages
print("Installing Python dependencies...")
os.system(f"pip install --no-cache-dir {' '.join(python_packages)}")


print("Installation completed. Try running your script again!")
