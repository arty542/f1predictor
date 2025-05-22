import pandas as pd
import numpy as np
from datetime import datetime

# Set seed for reproducibility
np.random.seed(42)

def load_data():
    """
    Load all required CSV files.
    """
    try:
        races = pd.read_csv('data/races.csv')
        drivers = pd.read_csv('data/drivers.csv')
        constructors = pd.read_csv('data/constructors.csv')
        results = pd.read_csv('data/results.csv')
        circuits = pd.read_csv('data/circuits.csv')
        return races, drivers, constructors, results, circuits
    except FileNotFoundError as e:
        print(f"Error: Could not find required CSV file - {e}")
        raise

def calculate_driver_experience(results):
    """
    Calculate driver experience as the number of prior races.
    """
    results = results.sort_values(by=["driverId", "raceId"])
    results["driver_experience"] = results.groupby("driverId").cumcount()
    return results

def calculate_constructor_experience(results):
    """
    Calculate constructor experience as the number of prior races.
    """
    results = results.sort_values(by=["constructorId", "raceId"])
    results["constructor_experience"] = results.groupby("constructorId").cumcount()
    return results

def add_driver_age(results, drivers, races):
    """
    Compute driver age in years at the time of race.
    """
    drivers = drivers[["driverId", "dob"]]
    races = races[["raceId", "date"]]

    merged = results.merge(drivers, on="driverId", how="inner")
    merged = merged.merge(races, on="raceId", how="inner")

    merged["dob"] = pd.to_datetime(merged["dob"], errors='coerce')
    merged["date"] = pd.to_datetime(merged["date"], errors='coerce')
    merged["driver_age"] = (merged["date"] - merged["dob"]).dt.days // 365
    return merged.drop(columns=["dob", "date"])

def add_constructor_performance(results):
    """
    Add constructor performance metrics.
    """
    # Calculate constructor's average position in last 5 races
    results = results.sort_values(by=["constructorId", "raceId"])
    results["constructor_avg_position"] = results.groupby("constructorId")["positionOrder"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    )
    return results

def add_driver_performance(results):
    """
    Add driver performance metrics.
    """
    # Calculate driver's average position in last 5 races
    results = results.sort_values(by=["driverId", "raceId"])
    results["driver_avg_position"] = results.groupby("driverId")["positionOrder"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    )
    return results

def encode_and_clean(df):
    """
    Clean and encode the dataset.
    """
    # Drop invalid entries
    df = df[(df["driverId"] != 0) & (df["constructorId"] != 0)]
    
    # Keep only top 20 positions
    df = df[df["positionOrder"] <= 20]
    
    # Select and order columns
    columns = [
        "driverId", "constructorId", "circuitId", "statusId", "grid",
        "driver_experience", "constructor_experience", "driver_age",
        "year", "positionOrder", "driver_avg_position", "constructor_avg_position"
    ]
    df = df[columns]
    
    # Fill NaN values with appropriate defaults
    df["driver_avg_position"] = df["driver_avg_position"].fillna(10)  # Middle of the pack
    df["constructor_avg_position"] = df["constructor_avg_position"].fillna(10)
    
    # Create mapping dictionaries for categorical columns
    # Use the original IDs as the mapping values
    driver_map = {id: id for id in df["driverId"].unique()}
    constructor_map = {id: id for id in df["constructorId"].unique()}
    circuit_map = {id: id for id in df["circuitId"].unique()}
    status_map = {id: id for id in df["statusId"].unique()}
    
    # Apply mappings
    df["driverId"] = df["driverId"].map(driver_map)
    df["constructorId"] = df["constructorId"].map(constructor_map)
    df["circuitId"] = df["circuitId"].map(circuit_map)
    df["statusId"] = df["statusId"].map(status_map)
    
    return df.dropna()

def main():
    print("Loading files...")
    races, drivers, constructors, results, circuits = load_data()

    print("Generating features...")
    results = calculate_driver_experience(results)
    results = calculate_constructor_experience(results)
    
    print("Adding performance metrics...")
    results = add_constructor_performance(results)
    results = add_driver_performance(results)

    print("Merging with races to get year and circuit info...")
    results = results.merge(races[["raceId", "year", "circuitId"]], on="raceId", how="inner")

    print("Adding driver age...")
    results = add_driver_age(results, drivers, races)

    print("Encoding & cleaning...")
    final_df = encode_and_clean(results)

    print("Saving to f1dataset.csv...")
    final_df.to_csv("f1dataset.csv", index=False)
    print("✅ Dataset created: f1dataset.csv")

if __name__ == "__main__":
    main()