# /app/core/config.py

import os

def load_env():
    if "ENV" not in os.environ:
        from dotenv import load_dotenv
        load_dotenv()