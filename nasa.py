import os
import requests
from dotenv import load_dotenv
from pathlib import Path
from utils import logger

load_dotenv(Path(__file__).parent / ".env")

USERNAME = os.getenv("EARTHDATA_USERNAME")
PASSWORD = os.getenv("EARTHDATA_PASSWORD")
session = requests.Session()

session.auth = (
    USERNAME,
    PASSWORD
)