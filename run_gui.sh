#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in the PATH."
    exit 1
fi

# Check if requirements are installed
if [ ! -f "requirements.txt" ]; then
    echo "Warning: requirements.txt not found."
else
    echo "Checking dependencies..."
    pip3 install -r requirements.txt
fi

# Run the GUI application
echo "Starting EEG Seizure Detection GUI..."
python3 eeg_seizure_gui.py 