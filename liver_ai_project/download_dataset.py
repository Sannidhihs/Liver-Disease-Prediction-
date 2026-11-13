"""Download the UCI ILPD dataset into data/ilpd.csv."""
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

out = Path(__file__).parent / "data" / "ilpd.csv"
out.parent.mkdir(parents=True, exist_ok=True)
dataset = fetch_ucirepo(id=225)
df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")
