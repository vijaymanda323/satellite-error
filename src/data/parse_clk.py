"""
parse_clk.py

Production-quality parser for converting decompressed CDDIS Precise Clock (CLK) files
into a unified, validated pandas DataFrame and exporting to CSV.

Leverages `georinex` for standard parsing and `pandas` for high-performance dataset merging.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import georinex as gr
from loguru import logger


class ClkParser:
    """
    Parses precise clock files (.clk) and extracts high-rate clock bias data.
    """

    def __init__(self, input_dir: str = "/data/raw/clk", output_file: str = "/data/processed/clk.csv"):
        """
        Initializes the parser.
        
        Args:
            input_dir (str): Directory containing uncompressed CLK files.
            output_file (str): Target path for the final CSV.
        """
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def parse_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Parses a single CLK file natively without georinex to avoid strict extension failures.

        Args:
            file_path (Path): Path to the CLK file.

        Returns:
            pd.DataFrame: Parsed dataframe or None if parsing fails.
        """
        try:
            logger.info(f"Parsing CLK file: {file_path.name}")
            
            actual_path = file_path
            
            with open(file_path, "rb") as f:
                magic = f.read(2)
            
            is_compressed = (magic == b'\x1f\x9d')
            is_gz = (magic == b'\x1f\x8b')
            
            if file_path.suffix == ".Z" and not is_compressed:
                logger.info(f"File {file_path.name} is already uncompressed. Dropping .Z extension.")
                new_path = file_path.with_suffix("")
                file_path.rename(new_path)
                actual_path = new_path
            
            import gzip
            open_fn = gzip.open if is_gz or actual_path.suffix == ".gz" else open
            
            records = []
            with open_fn(actual_path, "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("AS "):
                        # Format: AS G01  2026 01 01 00 00 00.000000  2   0.000123456789
                        parts = line.split()
                        if len(parts) >= 10:
                            sat_id = parts[1]
                            year, month, day, hour, minute, second = parts[2:8]
                            bias = float(parts[9])
                            
                            # Standardize to Timestamp
                            # Handle seconds > 60 if leap second
                            sec = float(second)
                            if sec >= 60: sec = 59.999999
                                
                            epoch = pd.Timestamp(year=int(year), month=int(month), day=int(day),
                                                 hour=int(hour), minute=int(minute), 
                                                 second=int(sec), microsecond=int((sec % 1) * 1e6))
                            
                            records.append({
                                "epoch": epoch,
                                "satellite_id": sat_id,
                                "clock_bias": bias
                            })
                            
            if not records:
                logger.warning(f"No valid clock bias records found in {actual_path.name}")
                return None
                
            df = pd.DataFrame(records)
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path.name}. Error: {e}")
            return None

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the merged dataframe by sorting and dropping duplicates.

        Args:
            df (pd.DataFrame): The combined dataframe.

        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        if df.empty:
            logger.warning("Dataframe is empty after parsing.")
            return df

        initial_count = len(df)
        df = df.drop_duplicates(subset=["epoch", "satellite_id"])
        df = df.sort_values(by=["epoch", "satellite_id"])
        
        final_count = len(df)
        logger.info(f"Validation complete: removed {initial_count - final_count} duplicate records.")
        return df

    def parse_directory(self) -> pd.DataFrame:
        """
        Iterates over the input directory, parses all files, and merges them.

        Returns:
            pd.DataFrame: Merged and validated dataframe.
        """
        all_dataframes = []
        
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            return pd.DataFrame()

        files = [f for f in self.input_dir.iterdir() if f.is_file()]
        
        for file_path in files:
            df = self.parse_file(file_path)
            if df is not None and not df.empty:
                all_dataframes.append(df)
            else:
                logger.warning(f"Skipping {file_path.name}")

        if not all_dataframes:
            logger.error("No valid data parsed from CLK files.")
            return pd.DataFrame()

        logger.info(f"Merging {len(all_dataframes)} CLK files...")
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        return self.validate_dataframe(merged_df)

    def save_csv(self, df: pd.DataFrame) -> None:
        """
        Saves the dataframe to CSV.
        """
        if df.empty:
            logger.warning("Cannot save empty dataframe.")
            return

        try:
            logger.info(f"Saving {len(df)} rows to {self.output_file}")
            df.to_csv(self.output_file, index=False)
            logger.success(f"Successfully exported CLK data to {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")


if __name__ == "__main__":
    parser = ClkParser()
    merged_data = parser.parse_directory()
    parser.save_csv(merged_data)
