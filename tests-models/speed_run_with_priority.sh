#!/bin/bash

# Run the Python script in the background
python tests/speed_cpu_tcn-tf.py &

# Get the PID of the last background process
PID=$!

# Increase priority of the script
sudo renice -n -20 -p $PID

# Wait for the script to finish
wait $PID
