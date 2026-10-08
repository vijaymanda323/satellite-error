"""
parse_rinex.py

Production-quality parser for converting decompressed CDDIS RINEX Navigation files
into a unified, validated pandas DataFrame and exporting to CSV.

Leverages `georinex` for standard parsing and `pandas` for high-performance dataset merging.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import georinex as gr
from loguru import logger


class RinexParser:
    """
    Parses RINEX Navigation files and extracts essential orbit and clock parameters.
    """

    def __init__(self, input_dir: str = "/data/raw/rinex", output_file: str = "/data/processed/broadcast.csv"):
        """
        Initializes the parser.
        
        Args:
            input_dir (str): Directory containing uncompressed RINEX files.
            output_file (str): Target path for the final CSV.
        """
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Expected column mapping from georinex xarray dataset to our CSV format
        self.column_mapping = {
            "SVclockBias": "clock_bias",
            "SVclockDrift": "clock_drift",
            "SVclockDriftRate": "clock_drift_rate",
            "sqrtA": "sqrtA",
            "Eccentricity": "eccentricity",
            "Io": "inclination",
            "Omega0": "raan",
            "omega": "argument_of_perigee",
            "M0": "mean_anomaly",
            "Toe": "toe",
            "SVhealth": "health",
            "SVacc": "accuracy",
            "GPSWeek": "week"
        }

    def parse_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Parses a single RINEX navigation file.

        Args:
            file_path (Path): Path to the RINEX file.

        Returns:
            pd.DataFrame: Parsed dataframe or None if parsing fails.
        """
        try:
            logger.info(f"Parsing RINEX file: {file_path.name}")
            
            actual_path = file_path
            
            # Check if the file is genuinely compressed by reading LZW magic bytes
            with open(file_path, "rb") as f:
                magic = f.read(2)
            
            is_compressed = (magic == b'\x1f\x9d')
            
            # If the user previously decompressed the file but left the .Z extension,
            # georinex will crash. We rename it to drop the .Z extension.
            if file_path.suffix == ".Z" and not is_compressed:
                logger.info(f"File {file_path.name} is already uncompressed. Dropping .Z extension.")
                new_path = file_path.with_suffix("")
                file_path.rename(new_path)
                actual_path = new_path

            # georinex parses into an xarray dataset.
            # We explicitly exclude IRNSS (I) and SBAS (S) as their non-standard fields crash the parser.
            ds = gr.load(actual_path, use=["G", "R", "E", "C"])
            
            # Convert xarray dataset to pandas dataframe
            df = ds.to_dataframe().reset_index()
            
            # Standardize column names
            df = df.rename(columns={"time": "epoch", "sv": "satellite_id"})
            
            # Extract requested columns if they exist in the file (RINEX 2/3 variations)
            available_columns = ["epoch", "satellite_id"]
            
            # Map standard columns
            for rinex_col, custom_col in self.column_mapping.items():
                if rinex_col in df.columns:
                    df = df.rename(columns={rinex_col: custom_col})
                    available_columns.append(custom_col)
                elif custom_col in df.columns:
                    # In some versions of georinex, it might already be lowercased
                    available_columns.append(custom_col)
                else:
                    logger.debug(f"Column '{rinex_col}' missing in {file_path.name}")
            
            # Filter only the extracted columns that were found
            valid_cols = [c for c in available_columns if c in df.columns]
            df = df[valid_cols]
            
            # Drop rows where critical data (like clock_bias) is completely NaN
            if "clock_bias" in df.columns:
                df = df.dropna(subset=["clock_bias", "epoch", "satellite_id"], how="any")
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path.name}. Error: {e}")
            return None

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the merged dataframe by sorting, dropping duplicates, and checking integrity.

        Args:
            df (pd.DataFrame): The combined dataframe.

        Returns:
            pd.DataFrame: The cleaned and validated dataframe.
        """
        if df.empty:
            logger.warning("Dataframe is empty after parsing.")
            return df

        # Remove complete duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=["epoch", "satellite_id"])
        
        # Sort chronologically
        df = df.sort_values(by=["epoch", "satellite_id"])
        
        final_count = len(df)
        logger.info(f"Validation complete: removed {initial_count - final_count} duplicate records.")
        return df

    def parse_directory(self) -> pd.DataFrame:
        """
        Iterates over the input directory, parses all files, and merges them.

        Returns:
            pd.DataFrame: The fully merged and validated dataframe.
        """
        all_dataframes = []
        
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            return pd.DataFrame()

        # Gather all files (ignoring subdirectories for simplicity)
        files = [f for f in self.input_dir.iterdir() if f.is_file()]
        
        if not files:
            logger.warning(f"No files found in {self.input_dir}")
            return pd.DataFrame()

        for file_path in files:
            df = self.parse_file(file_path)
            if df is not None and not df.empty:
                all_dataframes.append(df)
            else:
                logger.warning(f"Skipping {file_path.name} - empty or corrupted.")

        if not all_dataframes:
            logger.error("No valid data parsed from any file.")
            return pd.DataFrame()

        # Merge all dataframes
        logger.info(f"Merging {len(all_dataframes)} files...")
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Validate
        validated_df = self.validate_dataframe(merged_df)
        return validated_df

    def save_csv(self, df: pd.DataFrame) -> None:
        """
        Saves the resulting dataframe to a CSV file.

        Args:
            df (pd.DataFrame): Dataframe to save.
        """
        if df.empty:
            logger.warning("Cannot save empty dataframe. Aborting.")
            return

        try:
            logger.info(f"Saving {len(df)} rows to {self.output_file}")
            df.to_csv(self.output_file, index=False)
            logger.success(f"Successfully exported data to {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")


if __name__ == "__main__":
    parser = RinexParser()
    merged_data = parser.parse_directory()
    parser.save_csv(merged_data)
