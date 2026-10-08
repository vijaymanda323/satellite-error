import pandas as pd
import numpy as np
from loguru import logger

def main():
    logger.info("Loading broadcast and clock data...")
    try:
        df_brdc = pd.read_csv("broadcast.csv", parse_dates=["epoch"])
        df_clk = pd.read_csv("clk.csv", parse_dates=["epoch"])
    except FileNotFoundError as e:
        logger.error(f"Missing required CSV files: {e}")
        return

    logger.info(f"Loaded {len(df_brdc)} broadcast records and {len(df_clk)} precise clock records.")
    
    # Sort by epoch for merge_asof
    df_brdc = df_brdc.sort_values(by="epoch")
    df_clk = df_clk.sort_values(by="epoch")
    
    # Clean satellite IDs in broadcast (sometimes they have suffixes like _1)
    df_brdc['satellite_id'] = df_brdc['satellite_id'].astype(str).str.split('_').str[0].astype(str)
    df_clk['satellite_id'] = df_clk['satellite_id'].astype(str)
    
    logger.info("Merging broadcast and precise clock data...")
    # Merge on time, matching on satellite_id.
    # Broadcast ephemerides are valid into the future (typically +/- 2 hours from 'toe').
    # We forward-fill the broadcast ephemeris (direction='backward' means for each exact time in df_clk, 
    # we look backward in df_brdc for the most recent message).
    # We set a tolerance of 2 hours.
    df = pd.merge_asof(
        df_clk, 
        df_brdc, 
        on="epoch", 
        by="satellite_id", 
        direction="backward",
        tolerance=pd.Timedelta(hours=2)
    )
    
    # Drop rows where we couldn't find a valid broadcast message
    initial_len = len(df)
    df = df.dropna(subset=['clock_bias_y']) # 'clock_bias_y' is the broadcast bias
    logger.info(f"Matched {len(df)} precise readings with broadcast data (Dropped {initial_len - len(df)} rows due to missing broadcast data in the 2-hour window).")
    
    logger.info("Engineering features...")
    # Target Variable: Clock Error
    # clock_bias_x is from clk.csv (precise), clock_bias_y is from broadcast.csv
    df['clock_error'] = df['clock_bias_x'] - df['clock_bias_y']
    
    # Time-based features
    # Cyclical hour of day
    df['hour'] = df['epoch'].dt.hour + df['epoch'].dt.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    # Day of year
    df['day_of_year'] = df['epoch'].dt.dayofyear
    
    # Constellation
    # G = GPS, E = Galileo, R = GLONASS, C = BDS
    df['constellation'] = df['satellite_id'].str[0]
    
    # Drop redundant columns
    df = df.rename(columns={
        'clock_bias_x': 'precise_clock_bias',
        'clock_bias_y': 'broadcast_clock_bias'
    })
    
    # Reorder columns
    cols = ['epoch', 'satellite_id', 'constellation', 'clock_error', 
            'precise_clock_bias', 'broadcast_clock_bias', 
            'clock_drift', 'clock_drift_rate',
            'hour', 'hour_sin', 'hour_cos', 'day_of_year',
            'sqrtA', 'eccentricity', 'inclination', 'raan', 
            'argument_of_perigee', 'mean_anomaly', 'toe', 'health', 'accuracy']
    
    # only keep columns that exist in the dataframe
    cols_to_keep = [c for c in cols if c in df.columns]
    df = df[cols_to_keep]
    
    logger.info("Saving features.csv...")
    df.to_csv("features.csv", index=False)
    logger.info(f"Done. Final shape: {df.shape}")
    print("\nSample of final dataset:")
    print(df[['epoch', 'satellite_id', 'clock_error']].head())

if __name__ == "__main__":
    main()
