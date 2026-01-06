import uvicorn
import os
import sys

# Load environment variables
try:
    from dotenv import load_dotenv
    backend_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    root_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    
    if os.path.exists(backend_env):
        print(f"Loading .env from {backend_env}")
        load_dotenv(backend_env)
    elif os.path.exists(root_env):
        print(f"Loading .env from {root_env}")
        load_dotenv(root_env)
    else:
        print("Warning: No .env file found in backend or root directory.")
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables from .env will not be loaded.")

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

if __name__ == "__main__":
    print(f"Starting server from {BACKEND_DIR}")
    try:
        uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True, reload_dirs=[BACKEND_DIR])
    except Exception as e:
        print(f"Failed to start uvicorn: {e}")
