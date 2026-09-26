"""
app.py
------
Convenience launcher. Streamlit apps must be started via the `streamlit`
CLI (not `python dashboard.py`), so this script just does that for you
and gives a clear error if Streamlit isn't installed.

Usage:
    python app.py
"""

import subprocess
import sys
import os


def main():
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path], check=True)
    except FileNotFoundError:
        print("Streamlit is not installed. Run: pip install -r requirements.txt")
    except subprocess.CalledProcessError as e:
        print(f"Dashboard exited with an error: {e}")


if __name__ == "__main__":
    main()
