import modal
from datetime import date
from src.data.download import Downloader

app = modal.App("satellite-error-prediction")
volume = modal.Volume.from_name("satellite-data-vol")

# Include packages and explicitly add the src directory to the Image
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("src", remote_path="/root/src")
)

@app.function(
    image=image,
    volumes={"/data": volume},
    # Pass local .env variables to the Modal container securely
    secrets=[modal.Secret.from_dotenv()] 
)
def run_downloader():
    downloader = Downloader()
    # Use a historical date because NASA takes weeks to release "final" precise orbit files (igs*).
    # If we use today's date, the files do not exist yet and NASA returns a 404 Not Found error.
    target_date = date(2026, 1, 1)  # Jan 1, 2026
    downloader.download_day(target_date)
    downloader.generate_report()
@app.function(
    image=image,
    volumes={"/data": volume}
)
def run_parsers():
    # Parse RINEX
    from src.data.parse_rinex import RinexParser
    r_parser = RinexParser()
    r_parser.save_csv(r_parser.parse_directory())
    
    # Parse SP3
    from src.data.parse_sp3 import Sp3Parser
    s_parser = Sp3Parser()
    s_parser.save_csv(s_parser.parse_directory())
    
    # Parse CLK
    from src.data.parse_clk import ClkParser
    c_parser = ClkParser()
    c_parser.save_csv(c_parser.parse_directory())
