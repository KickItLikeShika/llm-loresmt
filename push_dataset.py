import pandas as pd
from datasets import Dataset

synthetic_data_path = "data/syntheticdata.csv"
df_synthetic = pd.read_csv(synthetic_data_path)
dataset = Dataset.from_pandas(df_synthetic)

repo_id = "KickItLikeShika/english-tatar-translation"
print(f"pushing to HF: {repo_id}...")
dataset.push_to_hub(repo_id)

