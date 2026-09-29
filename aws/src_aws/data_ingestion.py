import pandas as pd
import os

class DataIngestion:

    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self) -> pd.DataFrame:

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"File is not found!: {self.filepath}"
            )

        df = pd.read_csv(self.filepath)

        print(f"Dataset loaded successfully")
        print(f"Shape: {df.shape}")
        print(f"Features : {len(df.columns)}")

        return df
    
if __name__ == "__main__":

    print("Running Data Ingestion Module...")
    
    loader = DataIngestion("data/data_B.csv")
    df = loader.load_data()

    print(df.head())