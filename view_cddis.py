import os
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv(".env")

def view_directory():
    url = "https://cddis.nasa.gov/archive/gnss/data/daily/2026/001/"
    session = requests.Session()
    resp = session.get(url)
    
    if resp.status_code == 200:
        lines = resp.text.split("\n")
        # Find links in the HTML that look like files or directories
        links = [line for line in lines if ("href=" in line.lower()) and ("26" in line.lower() or ".z" in line.lower() or ".gz" in line.lower())]
        for l in links:
            print(l.strip())
    else:
        print(f"Failed with {resp.status_code}")

if __name__ == "__main__":
    view_directory()
