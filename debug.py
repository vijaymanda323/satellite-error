import modal

app = modal.App("debug-app")
vol = modal.Volume.from_name("satellite-data-vol")

@app.function(volumes={"/data": vol})
def check_file():
    with open('/data/raw/rinex/brdc1870.26p.Z', 'rb') as f:
        data = f.read(500)
        print(data)
