"""
parse_sp3.py

Production-quality parser for converting decompressed CDDIS Precise Orbit (SP3) files
into a unified, validated pandas DataFrame and exporting to CSV.

Leverages `georinex` for standard parsing and `pandas` for high-performance dataset merging.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import georinex as gr
from loguru import logger


class Sp3Parser:
    """
    Parses SP3 Precise Orbit files and extracts position and clock values.
    """

    def __init__(self, input_dir: str = "/data/raw/sp3", output_file: str = "/data/processed/sp3.csv"):
        """
        Initializes the parser.
        
        Args:
            input_dir (str): Directory containing uncompressed SP3 files.
            output_file (str): Target path for the final CSV.
        """
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def parse_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Parses a single SP3 file.

        Args:
            file_path (Path): Path to the SP3 file.

        Returns:
            pd.DataFrame: Parsed dataframe or None if parsing fails.
        """
        try:
            logger.info(f"Parsing SP3 file: {file_path.name}")
            
            actual_path = file_path
            
            with open(file_path, "rb") as f:
                magic = f.read(2)
            
            is_compressed = (magic == b'\x1f\x9d')
            
            if file_path.suffix == ".Z" and not is_compressed:
                logger.info(f"File {file_path.name} is already uncompressed. Dropping .Z extension.")
                new_path = file_path.with_suffix("")
                file_path.rename(new_path)
                actual_path = new_path
            
            # georinex handles sp3 perfectly and returns an xarray dataset
            ds = gr.load(actual_path)
            
            # Convert xarray dataset to pandas dataframe
            df = ds.to_dataframe().reset_index()
            
            # SP3 parsed data usually has ['time', 'sv', 'ECEF', 'position', 'clock']
            # We want to pivot the ECEF coordinate (x,y,z) into separate columns if it's long format
            # Depending on georinex version, position might be a vector. 
            
            # If the dataframe has an 'ECEF' dimension column (like 'x', 'y', 'z')
            if "ECEF" in df.columns and "position" in df.columns:
                # Pivot to wide format: x, y, z as columns
                df = df.pivot_table(
                    index=['time', 'sv'], 
                    columns='ECEF', 
                    values='position'
                ).reset_index()
                
                # Retrieve the clock if it exists by merging back or just joining
                # Clock is usually independent of ECEF, so we can extract it before pivot
                clock_df = ds['clock'].to_dataframe().reset_index()
                if "clock" in clock_df.columns:
                    # clock_df has ['time', 'sv', 'clock']
                    df = pd.merge(df, clock_df, on=['time', 'sv'], how='left')

            # Standardize names
            df = df.rename(columns={"time": "epoch", "sv": "satellite_id"})
            
            # Ensure columns exist and rename cleanly
            rename_map = {"x": "x", "y": "y", "z": "z"} 
            df = df.rename(columns=rename_map)
            
            # Keep standard columns if they exist
            cols_to_keep = ["epoch", "satellite_id"]
            for opt_col in ["x", "y", "z", "clock"]:
                if opt_col in df.columns:
                    cols_to_keep.append(opt_col)
                    
            df = df[cols_to_keep]
            
            # Drop empty records
            df = df.dropna(subset=["epoch", "satellite_id"])
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
            logger.error("No valid data parsed from SP3 files.")
            return pd.DataFrame()

        logger.info(f"Merging {len(all_dataframes)} SP3 files...")
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
            logger.success(f"Successfully exported SP3 data to {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")


if __name__ == "__main__":
    parser = Sp3Parser()
    merged_data = parser.parse_directory()
    parser.save_csv(merged_data)
