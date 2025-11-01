#test_env.py
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

print("HUMAI_ENV:", os.getenv("HUMAI_ENV"))
print("HUMAI_API_URL:", os.getenv("HUMAI_API_URL"))
print("HUMAI_REPORTS_DIR", os.getenv("HUMAI_REPORTS_DIR"))
