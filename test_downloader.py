from src.data.download import Downloader
from datetime import date

if __name__ == "__main__":
    downloader = Downloader(timeout=60)
    downloader.download_day(date(2026, 1, 1))
