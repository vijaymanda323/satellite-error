import requests

def test_cddis():
    url = "https://cddis.nasa.gov/archive/gnss/data/daily/2026/190/26p/brdc1900.26p.Z"
    
    # We will test without auth first
    resp = requests.get(url, allow_redirects=False)
    print(f"Initial Status Code: {resp.status_code}")
    print(f"Headers: {resp.headers}")
    
    if resp.status_code == 302:
        print(f"Redirects to: {resp.headers.get('Location')}")
        
if __name__ == "__main__":
    test_cddis()
