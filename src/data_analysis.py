from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "PlantSeg" / "Metadata.csv"


def load_metadata():
    """Load PlantSeg metadata."""
    df = pd.read_csv(METADATA_PATH)
    return df


def basic_analysis(df):
    """Display basic dataset information."""

    print("=" * 60)
    print("AGRISENSE - DATASET OVERVIEW")
    print("=" * 60)

    print(f"\nNumber of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nPlant distribution:")
    print(df["Plant"].value_counts())

    print("\nDisease distribution:")
    print(df["Disease"].value_counts())

    print("\nDataset split:")
    print(df["Split"].value_counts())

    print("\nDisease class statistics:")

    disease_counts = df["Disease"].value_counts()

    print(f"Total disease classes: {len(disease_counts)}")
    print(f"Smallest class: {disease_counts.min()} images")
    print(f"Largest class: {disease_counts.max()} images")
    print(f"Average images per class: {disease_counts.mean():.2f}")

    print("\nClasses with fewer than 20 images:")
    print(disease_counts[disease_counts < 20])


if __name__ == "__main__":
    metadata = load_metadata()
    basic_analysis(metadata)