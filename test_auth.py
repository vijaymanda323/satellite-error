import os
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv(".env")

def test_auth():
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")
    
    netrc_path = Path.home() / ".netrc"
    netrc_content = f"machine urs.earthdata.nasa.gov login {username} password {password}\n"
    netrc_content += f"machine cddis.nasa.gov login {username} password {password}\n"
    netrc_path.write_text(netrc_content)
    try:
        netrc_path.chmod(0o600)
    except: pass
    
    url_Z = "https://cddis.nasa.gov/archive/gnss/data/daily/2026/001/26p/brdc0010.26p.Z"
    url_gz = "https://cddis.nasa.gov/archive/gnss/data/daily/2026/001/26p/brdc0010.26p.gz"
    
    session = requests.Session()
    resp_Z = session.head(url_Z)
    resp_gz = session.head(url_gz)
    
    print(f"Status for .Z: {resp_Z.status_code}")
    print(f"Status for .gz: {resp_gz.status_code}")
    print(f"Status Code: {resp.status_code}")
    print(f"URL: {resp.url}")
    print(f"Text snippet: {resp.text[:200]}")

if __name__ == "__main__":
    test_auth()
