#!/usr/bin/env python3
"""
MicroLive Spot Simulation - CLI Entry Point
============================================

Run simulations from the command line.

Usage:
    python run_simulation.py
    python run_simulation.py --config my_config.yaml
    python run_simulation.py --config my_config.yaml --output ./results
"""

from spot_simulator import main

if __name__ == '__main__':
    main()
