from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
TEST_DIR = ROOT / "reviewer_test"

FEATURES = ["Row_Spacing", "Plant_Spacing"]
VARIETIES = ["DN251", "DN252", "DN253", "HN48", "HN51"]

def main():
    inputs = pd.read_csv(TEST_DIR / "reviewer_test_inputs.csv")
    expected = pd.read_csv(TEST_DIR / "expected_predictions.csv")

    outputs = []
    all_passed = True

    print("=" * 68)
    print("Plant Phenomics reviewer model test")
    print("=" * 68)

    for variety in VARIETIES:
        model_path = MODEL_DIR / f"{variety}_rf_surrogate.joblib"
        model = joblib.load(model_path)

        subset = inputs[inputs["Variety"] == variety].copy()
        pred = model.predict(subset[FEATURES])

        exp_subset = expected[expected["Variety"] == variety].copy()
        exp_subset = exp_subset.sort_values("Treatment_Index")
        exp = exp_subset["Expected_RF_Yield_Score"].to_numpy()

        passed = np.allclose(pred, exp, rtol=1e-8, atol=1e-10)
        all_passed = all_passed and passed

        print(f"{variety}: model loaded; {len(subset)} test rows; "
              f"status={'PASS' if passed else 'FAIL'}")

        for (_, row), p in zip(subset.iterrows(), pred):
            outputs.append({
                "Variety": variety,
                "Treatment_Index": int(row["Treatment_Index"]),
                "Row_Spacing": float(row["Row_Spacing"]),
                "Plant_Spacing": float(row["Plant_Spacing"]),
                "Observed_True_Yield": row["Observed_True_Yield"],
                "Predicted_RF_Yield_Score": float(p),
            })

    out = pd.DataFrame(outputs)
    out.to_csv(ROOT / "reviewer_predictions.csv", index=False)

    print("-" * 68)
    if all_passed:
        print("All reviewer tests passed.")
        print("Output written to reviewer_predictions.csv")
        return 0

    print("One or more reviewer tests failed.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
