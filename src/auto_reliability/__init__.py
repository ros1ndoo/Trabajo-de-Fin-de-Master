"""Auto Reliability: a transparent, recall-based safety proxy for US vehicles.

The project deliberately distinguishes a NHTSA recall proxy from general
mechanical reliability.  Public modules are designed to be usable both from
the command line and from the Streamlit dashboard.
"""

from .config import ProjectPaths

__all__ = ["ProjectPaths"]
__version__ = "0.1.0"
