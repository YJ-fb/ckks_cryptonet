#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

"""
This file initializes the package and makes its components available for import.
"""

# Import submodules to make them accessible when importing the main module
from .cryptonet import Cryptonet
from .encryption import Encryption
from .dataset import Dataset
from .debug import Debug
from .encryptednet import EncryptedNet
from .plainnet import PlainNet
from .encodenet import EncodeNet
from .exporter import Exporter

# Optionally, you can also expose specific classes or functions directly from this file
# For example:
# from .cryptonet import Cryptonet
# from .encryption import Encryption

# If there are any constants or utility functions that should be available at the package level,
# they can be imported here as well.

# Initialize package-level variables or configurations if necessary
# For instance, setting up logging, initializing global settings, etc.
# This section is optional and depends on the requirements of your application

# Example of initializing a verbosity flag (if used across multiple modules)
verbosity = False  # This can be set by users or through command-line arguments

# Expose version information
# Assuming each submodule has a __version__ attribute
__version__ = "0.1.0"  # Replace with actual versioning strategy

def set_verbosity(flag):
    """Set the verbosity flag for all modules in the package.

    Args:
        flag (bool): Whether to enable verbose output.
    """
    global verbosity
    verbosity = flag
    
    # Propagate the verbosity setting to all relevant modules
    Cryptonet.set_verbosity(flag)
    Encryption.set_verbosity(flag)
    Dataset.set_verbosity(flag)
    Debug.set_verbosity(flag)
    EncryptedNet.set_verbosity(flag)
    PlainNet.set_verbosity(flag)
    EncodeNet.set_verbosity(flag)
    Exporter.set_verbosity(flag)

# Optional: Provide a function to initialize the entire package
def init_package():
    """Initialize the package with default settings."""
    # Here you can perform any initialization steps required for the package
    # For example, setting up directories, loading configurations, etc.
    pass

# Optional: Provide a cleanup function if resources need to be released
def cleanup_package():
    """Cleanup resources used by the package."""
    # Here you can define steps to clean up resources when the program ends
    pass

# Optionally, you can provide helper functions or utilities at the package level
# These could be functions that combine functionality from multiple submodules
