#!/bin/bash

# Run the placebo_0 script
echo "Running placebo_0 script..."
python3 ./placebo_0/placebo_discrimination_analyses.py

# Run the placebo_10 script
echo "Running placebo_10 script..."
python3 ./placebo_10/placebo_discrimination_analyses.py

echo "Both scripts finished!"