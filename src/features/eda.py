import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("Loading features.csv...")
    df = pd.read_csv("features.csv", parse_dates=["epoch"])
    
    # Create output directory for plots
    os.makedirs("eda_plots", exist_ok=True)
    
    print("Generating EDA plots...")
    
    # 1. Distribution of Clock Errors
    plt.figure(figsize=(10, 6))
    # Filter extreme outliers for better visualization (e.g., within 99th percentile)
    q_low = df["clock_error"].quantile(0.01)
    q_hi  = df["clock_error"].quantile(0.99)
    df_filtered = df[(df["clock_error"] < q_hi) & (df["clock_error"] > q_low)]
    
    sns.histplot(df_filtered["clock_error"], bins=50, kde=True)
    plt.title("Distribution of Clock Errors (1st-99th Percentile)")
    plt.xlabel("Clock Error (seconds)")
    plt.ylabel("Frequency")
    plt.savefig("eda_plots/error_distribution.png")
    plt.close()

    # 2. Clock Error over Time by Constellation
    plt.figure(figsize=(12, 6))
    # Sample down to avoid huge plotting time if necessary, but 92k is okay for scatter
    sns.scatterplot(data=df_filtered, x="epoch", y="clock_error", hue="constellation", alpha=0.5, s=10)
    plt.title("Clock Error over Time by Constellation")
    plt.xlabel("Epoch")
    plt.ylabel("Clock Error (seconds)")
    plt.legend(title="Constellation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("eda_plots/error_over_time.png")
    plt.close()
    
    # 3. Boxplot of Clock Error by Constellation
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_filtered, x="constellation", y="clock_error")
    plt.title("Clock Error Distribution by Constellation")
    plt.xlabel("Constellation")
    plt.ylabel("Clock Error (seconds)")
    plt.savefig("eda_plots/error_boxplot.png")
    plt.close()
    
    # Generate summary stats
    stats = df.groupby('constellation')['clock_error'].describe()
    stats.to_csv("eda_plots/summary_stats.csv")
    print("EDA completed. Plots saved to eda_plots/ directory.")

if __name__ == "__main__":
    main()
