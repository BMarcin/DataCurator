import os
import sys

# The service modules (server.py, batcher.py) are flat, top-level modules — put
# this directory on sys.path so the tests can import them without a package.
sys.path.insert(0, os.path.dirname(__file__))
