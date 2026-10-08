"""
download.py

Production-quality downloader for downloading GNSS datasets (RINEX, SP3, CLK)
from NASA CDDIS. The module handles authentication, retry logic, file validation,
and reporting of successes and failures.

Future Upgrades:
The architecture is modular. Replacing `requests` with `aiohttp` and adding
async methods later will not break the public API.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import List, Dict, Any
import requests

import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# Import authentication and URL generation functions.
# Assuming nasa.py is located in the same module based on project structure.
from .nasa import get_session, build_rinex_url, build_sp3_url, build_clk_url


class DownloadError(Exception):
    """Custom exception raised for non-transient download failures or validation errors."""
    pass


class Downloader:
    """
    Manages the downloading of GNSS datasets from CDDIS.

    Responsibilities:
    - Maintains authenticated requests.Session.
    - Creates necessary folder structure automatically.
    - Validates file sizes and skips already downloaded files.
    - Retries transient errors using tenacity.
    - Logs all events to Loguru.
    - Generates success and failure reports using Pandas.
    """

    def __init__(self, output_dir: Path = Path("/data/raw"), timeout: int = 30, max_retries: int = 5):
        """
        Initializes the Downloader instance.

        Args:
            output_dir (Path): The root directory for storing downloaded datasets. Default is /data/raw.
            timeout (int): HTTP request timeout in seconds.
            max_retries (int): Unused directly here as Tenacity decorator is static, 
                               but kept for future dynamic retry configurations.
        """
        self.output_dir = output_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = get_session()
        
        # Internal state lists for generating Pandas DataFrame reports later
        self.successful_downloads: List[Dict[str, Any]] = []
        self.failed_downloads: List[Dict[str, Any]] = []
        
        # Ensure directories exist upon initialization
        self._create_directories()

    def _create_directories(self) -> None:
        """
        Creates the standard folder structure for the GNSS datasets.
        Creates /raw/rinex/, /raw/sp3/, and /raw/clk/
        """
        (self.output_dir / "rinex").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "sp3").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "clk").mkdir(parents=True, exist_ok=True)
        logger.info(f"Verified directory structure at {self.output_dir}")

    def verify(self, file_path: Path) -> bool:
        """
        Verifies if a downloaded file is valid.
        Currently checks if the file exists and is not empty.

        Args:
            file_path (Path): Path to the downloaded file.

        Returns:
            bool: True if the file is valid, False otherwise.
        """
        if not file_path.exists():
            return False
        
        # Validate that the file size is greater than 0 bytes
        if file_path.stat().st_size == 0:
            logger.warning(f"Validation failed: File {file_path.name} is empty (0 bytes).")
            return False
            
        return True

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2)
    )
    def _execute_download(self, url: str, output_path: Path) -> None:
        """
        Executes the HTTP GET request with robust retry logic for transient failures.
        Downloads data in chunks and displays a tqdm progress bar.

        Args:
            url (str): The target URL to download.
            output_path (Path): The local destination file path.

        Raises:
            requests.exceptions.RequestException: Upon network or HTTP failure, triggering tenacity retry.
        """
        # We don't catch exceptions here so that @retry can automatically catch them and retry.
        # Once retries are exhausted, the exception will propagate up.
        with self.session.get(url, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            
            with open(output_path, "wb") as f, tqdm(
                desc=output_path.name,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                leave=False
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        size = f.write(chunk)
                        pbar.update(size)

    def download_single_file(self, url: str, output_path: Path, file_type: str) -> bool:
        """
        Coordinates the downloading and verifying of a single file.
        Skips the download if the file is already downloaded and valid.

        Args:
            url (str): Remote URL of the file.
            output_path (Path): Local file path to save the content.
            file_type (str): Identifier for the dataset type (e.g., 'RINEX').

        Returns:
            bool: True if download is successful or file already exists, False otherwise.
        """
        if self.verify(output_path):
            logger.info(f"Skipping existing valid file: {output_path.name}")
            self.successful_downloads.append({
                "url": url,
                "file_path": str(output_path),
                "type": file_type,
                "status": "skipped"
            })
            return True
            
        logger.info(f"Starting download: {output_path.name} from {url}")
        try:
            self._execute_download(url, output_path)
            
            if self.verify(output_path):
                logger.success(f"Successfully downloaded: {output_path.name}")
                self.successful_downloads.append({
                    "url": url,
                    "file_path": str(output_path),
                    "type": file_type,
                    "status": "success"
                })
                return True
            else:
                raise DownloadError(f"File validation failed for {output_path.name} after download.")
                
        except Exception as e:
            logger.error(f"Download failed for {output_path.name}: {e}")
            self.failed_downloads.append({
                "url": url,
                "file_path": str(output_path),
                "type": file_type,
                "error": str(e)
            })
            
            # Clean up the partial or broken file if it exists
            if output_path.exists():
                logger.info(f"Cleaning up incomplete file: {output_path.name}")
                output_path.unlink()
                
            return False

    def download_rinex(self, target_date: date) -> bool:
        """
        Builds the URL and downloads the Broadcast Navigation File (RINEX) for a given date.

        Args:
            target_date (date): The date to download the data for.

        Returns:
            bool: True if successful, False otherwise.
        """
        url = build_rinex_url(target_date)
        file_name = url.split("/")[-1]
        output_path = self.output_dir / "rinex" / file_name
        return self.download_single_file(url, output_path, "RINEX")

    def download_sp3(self, target_date: date) -> bool:
        """
        Builds the URL and downloads the Precise Orbit File (SP3) for a given date.

        Args:
            target_date (date): The date to download the data for.

        Returns:
            bool: True if successful, False otherwise.
        """
        url = build_sp3_url(target_date)
        file_name = url.split("/")[-1]
        output_path = self.output_dir / "sp3" / file_name
        return self.download_single_file(url, output_path, "SP3")

    def download_clk(self, target_date: date) -> bool:
        """
        Builds the URL and downloads the Precise Clock File (CLK30S) for a given date.

        Args:
            target_date (date): The date to download the data for.

        Returns:
            bool: True if successful, False otherwise.
        """
        url = build_clk_url(target_date)
        file_name = url.split("/")[-1]
        output_path = self.output_dir / "clk" / file_name
        return self.download_single_file(url, output_path, "CLK")

    def download_day(self, target_date: date) -> None:
        """
        Downloads all essential GNSS datasets (RINEX, SP3, CLK) for a single day.

        Args:
            target_date (date): The date to download datasets for.
        """
        logger.info(f"--- Starting downloads for date: {target_date} ---")
        self.download_rinex(target_date)
        self.download_sp3(target_date)
        self.download_clk(target_date)
        logger.info(f"--- Completed downloads for date: {target_date} ---")

    def download_date_range(self, start_date: date, end_date: date) -> None:
        """
        Iterates over a date range and downloads datasets for each day.
        Automatically generates a report at the end of the operation.

        Args:
            start_date (date): The start of the date range (inclusive).
            end_date (date): The end of the date range (inclusive).
        """
        delta = end_date - start_date
        if delta.days < 0:
            logger.error("Invalid range: end_date must be on or after start_date.")
            return

        logger.info(f"Starting date range download: {start_date} to {end_date} ({delta.days + 1} days)")
        for i in range(delta.days + 1):
            current_date = start_date + timedelta(days=i)
            self.download_day(current_date)
            
        self.generate_report()

    def generate_report(self) -> None:
        """
        Exports the records of successful and failed downloads into CSV files.
        Uses Pandas to save 'download_report.csv' and 'failed_downloads.csv' 
        in the base output directory.
        """
        logger.info("Generating final download reports...")
        
        # Report for successful downloads
        if self.successful_downloads:
            success_df = pd.DataFrame(self.successful_downloads)
            success_path = self.output_dir / "download_report.csv"
            success_df.to_csv(success_path, index=False)
            logger.success(f"Saved success report to {success_path}")
        else:
            logger.info("No successful downloads to report.")
            
        # Report for failed downloads
        if self.failed_downloads:
            failed_df = pd.DataFrame(self.failed_downloads)
            failed_path = self.output_dir / "failed_downloads.csv"
            failed_df.to_csv(failed_path, index=False)
            logger.warning(f"Saved failure report to {failed_path}")
        else:
            logger.info("No failed downloads to report. Perfect run!")


if __name__ == "__main__":
    # Get the current date dynamically
    today = date.today()
    logger.info(f"Running Downloader for current date: {today}")
    
    # Initialize the downloader (default output is /data/raw)
    downloader = Downloader()
    
    # Download RINEX, SP3, and CLK for today
    downloader.download_day(today)
    
    # Generate CSV reports
    downloader.generate_report()
