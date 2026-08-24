from pathlib import Path
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
VARIETIES = ["DN251", "DN252", "DN253", "HN48", "HN51"]
FEATURES = ["Row_Spacing", "Plant_Spacing"]
TARGET = "Yield_Score"

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    rows = []

    for variety in VARIETIES:
        path = DATA_DIR / f"{variety}_all_simulation_with_50cm_transect.csv"
        df = pd.read_csv(path)

        X = df[FEATURES]
        y = df[TARGET]

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)

        out = MODEL_DIR / f"{variety}_rf_surrogate.joblib"
        joblib.dump(model, out, compress=3)

        rows.append({
            "Variety": variety,
            "Model_File": out.name,
            "Training_Rows": len(df),
            "Training_R2": model.score(X, y),
            "SHA256": hashlib.sha256(out.read_bytes()).hexdigest(),
        })
        print(f"{variety}: saved {out.name}")

    pd.DataFrame(rows).to_csv(
        MODEL_DIR / "refreeze_metadata.csv", index=False
    )

if __name__ == "__main__":
    main()
