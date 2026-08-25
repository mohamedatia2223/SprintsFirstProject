import sys
import os
import runpy

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Launch Streamlit chat interface
if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(__file__), "src", "app", "chatStreamlit.py")
    runpy.run_path(app_path, run_name="__main__")
