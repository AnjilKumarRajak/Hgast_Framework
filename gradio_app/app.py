import os
import sys

# Import root app logic
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False)
