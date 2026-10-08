from pathlib import Path

RAW_DIR = Path("/data/raw")

RINEX_DIR = RAW_DIR / "rinex"
SP3_DIR = RAW_DIR / "sp3"
CLK_DIR = RAW_DIR / "clk"

MAX_RETRIES = 5

TIMEOUT = 120