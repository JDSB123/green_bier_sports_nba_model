"""
Pick Tracking Module

Provides dynamic live tracking of predictions vs outcomes.
Records all predictions at generation time and validates against final scores.
"""

from .tracker import PickTracker

__all__ = ["PickTracker"]
