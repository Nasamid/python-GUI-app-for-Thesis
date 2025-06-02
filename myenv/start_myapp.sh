#!/bin/bash
# Change to the project directory.
cd /home/jr/myenv

# Activate the virtual environment.
source bin/activate

# Start the gallery management script in the background.
nohup ./manage_gallery.sh &

# Start the main application.
exec python app.py
