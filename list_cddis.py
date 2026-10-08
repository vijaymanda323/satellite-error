import os
from pathlib import Path
from dotenv import load_dotenv
import requests
import re

load_dotenv(".env")

def view_directory():
    url = "https://cddis.nasa.gov/archive/gnss/data/daily/2026/001/26p/"
    
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")
    
    session = requests.Session()
    # Basic auth works for CDDIS if we just try downloading something first or use netrc
    # Actually, Earthdata redirect is tricky. We'll rely on .netrc.
    
    resp = session.get(url)
    if resp.status_code == 200:
        links = re.findall(r'href=[\'"]?([^\'" >]+)', resp.text)
        print(f"Found {len(links)} links in 26p directory:")
        for link in links:
            if "26" in link or "brdc" in link.lower() or "BRDC" in link:
                print(link)
                
    url_sp3 = "https://cddis.nasa.gov/archive/gnss/products/2399/"
    resp2 = session.get(url_sp3)
    if resp2.status_code == 200:
        links2 = re.findall(r'href=[\'"]?([^\'" >]+)', resp2.text)
        print(f"\nFound {len(links2)} links in products 2399 directory:")
        for link in links2:
            if "igs" in link.lower() or "IGS" in link or "sp3" in link.lower() or "clk" in link.lower():
                print(link)

if __name__ == "__main__":
    view_directory()
