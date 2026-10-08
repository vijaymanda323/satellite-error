"""
nasa.py

Handles authentication with NASA Earthdata and provides URL builders
for NASA CDDIS GNSS datasets (RINEX, SP3, CLK).

This module manages the HTTP Session credentials and formats the strict
CDDIS archive URLs using GPS time calculations.
"""

import os
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv
from loguru import logger

# Automatically load the .env file from the project root
# Path traversal: src/data/nasa.py -> src/data -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def login() -> requests.Session:
    """
    Authenticates with NASA Earthdata using credentials from the environment.
    Creates a .netrc file to safely handle NASA's URS redirects.
    
    Returns:
        requests.Session: An authenticated session configured for CDDIS.
        
    Raises:
        ValueError: If EARTHDATA_USERNAME or EARTHDATA_PASSWORD are not set.
    """
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")
    
    if not username or not password:
        logger.error("NASA Earthdata credentials missing. Check .env file.")
        raise ValueError("EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set.")

    # Create a .netrc file in the home directory.
    # This is the official and most reliable way to authenticate with NASA CDDIS,
    # because it automatically re-attaches credentials during cross-domain redirects.
    netrc_path = Path.home() / ".netrc"
    
    try:
        netrc_content = f"machine urs.earthdata.nasa.gov login {username} password {password}\n"
        netrc_content += f"machine cddis.nasa.gov login {username} password {password}\n"
        netrc_path.write_text(netrc_content)
        
        # .netrc requires strict permissions on Unix systems
        if os.name != 'nt':
            netrc_path.chmod(0o600)
    except Exception as e:
        logger.warning(f"Could not write .netrc file securely: {e}")

    session = requests.Session()
    logger.info(f"NASA Earthdata session configured via .netrc for user: {username}")
    return session


def get_session() -> requests.Session:
    """
    Wrapper around login() to return an active requests.Session.
    Designed for dependency injection in downloader scripts.
    
    Returns:
        requests.Session: Authenticated session.
    """
    return login()


def _get_gps_week(target_date: date) -> int:
    """
    Calculates the GPS week for a given date.
    GPS time started on Sunday, January 6, 1980.
    
    Args:
        target_date (date): Date to convert.
        
    Returns:
        int: GPS Week number.
    """
    gps_epoch = date(1980, 1, 6)
    delta = target_date - gps_epoch
    return delta.days // 7


def _get_day_of_week(target_date: date) -> int:
    """
    Calculates the GPS day of week (0 = Sunday, 6 = Saturday).
    
    Args:
        target_date (date): Date to convert.
        
    Returns:
        int: Day of the week integer (0-6).
    """
    # target_date.isoweekday() returns 1 (Monday) to 7 (Sunday)
    # We want 0 (Sunday) to 6 (Saturday)
    return target_date.isoweekday() % 7


def build_rinex_url(target_date: date) -> str:
    """
    Builds the CDDIS URL for the Broadcast Navigation File (RINEX).
    Uses the modern IGS long filename convention (mandatory for data after 2022).
    
    Format: https://cddis.nasa.gov/archive/gnss/data/daily/YYYY/DDD/YYp/BRDC00IGS_R_YYYYDDD0000_01D_MN.rnx.gz
    
    Args:
        target_date (date): Date of the dataset.
        
    Returns:
        str: Fully qualified download URL.
    """
    year = target_date.year
    yy = str(year)[-2:]
    # Day of year padded to 3 digits (001 - 366)
    doy = target_date.strftime("%j")
    
    # IGS Long Filename for Mixed Broadcast Ephemeris
    file_name = f"BRDC00IGS_R_{year}{doy}0000_01D_MN.rnx.gz"
    base_url = f"https://cddis.nasa.gov/archive/gnss/data/daily/{year}/{doy}/{yy}p"
    return f"{base_url}/{file_name}"


def build_sp3_url(target_date: date) -> str:
    """
    Builds the CDDIS URL for the Precise Orbit File (SP3).
    Uses the modern IGS long filename convention for final orbits.
    
    Format: https://cddis.nasa.gov/archive/gnss/products/WWWW/IGS0OPSFIN_YYYYDDD0000_01D_15M_ORB.SP3.gz
    
    Args:
        target_date (date): Date of the dataset.
        
    Returns:
        str: Fully qualified download URL.
    """
    year = target_date.year
    doy = target_date.strftime("%j")
    gps_week = _get_gps_week(target_date)
    
    # IGS Long Filename for Final Orbit
    file_name = f"IGS0OPSFIN_{year}{doy}0000_01D_15M_ORB.SP3.gz"
    base_url = f"https://cddis.nasa.gov/archive/gnss/products/{gps_week}"
    return f"{base_url}/{file_name}"


def build_clk_url(target_date: date) -> str:
    """
    Builds the CDDIS URL for the Precise Clock File (CLK).
    Uses the modern IGS long filename convention for final clocks.
    
    Format: https://cddis.nasa.gov/archive/gnss/products/WWWW/IGS0OPSFIN_YYYYDDD0000_01D_30S_CLK.CLK.gz
    
    Args:
        target_date (date): Date of the dataset.
        
    Returns:
        str: Fully qualified download URL.
    """
    year = target_date.year
    doy = target_date.strftime("%j")
    gps_week = _get_gps_week(target_date)
    
    # IGS Long Filename for Final Clock
    file_name = f"IGS0OPSFIN_{year}{doy}0000_01D_30S_CLK.CLK.gz"
    base_url = f"https://cddis.nasa.gov/archive/gnss/products/{gps_week}"
    return f"{base_url}/{file_name}"
