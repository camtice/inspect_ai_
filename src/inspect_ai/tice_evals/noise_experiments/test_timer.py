import sys
import os
import time

# Set the INSPECT_DISPLAY environment variable to ensure timers are shown
# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath("src"))

# Import the test_timing function from the noise_hf module
from inspect_ai.model._providers.noise_hf import test_timing

# Run the test_timing function
if __name__ == "__main__":
    print("Testing timer functionality...")
    test_timing()
    print("Timer test complete.") 