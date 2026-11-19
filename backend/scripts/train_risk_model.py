import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings
from pipeline.preprocessing import FEATURE_COLUMNS, get_dataframe

BANDS = {"low_max": 0.20, "high_min": 0.45}


def band(probability: float) -> str:
    if probability >= BANDS["high_min"]:
        return "High"
    if probability >= BANDS["low_max"]:
        return "Medium"
    return "Low"


def data_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the deterministic diabetes risk model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=get_settings().risk_model_path)
    args = parser.parse_args()

    df = get_dataframe()
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["Outcome"].to_numpy(dtype=int)
    ids = df["Patient_ID"].to_numpy()

    X_train, X_test, y_train, y_test, _, ids_test = train_test_split(
        X, y, ids, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv_auc = cross_val_score(
        pipeline, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=args.seed), scoring="roc_auc"
    )
    pipeline.fit(X_train, y_train)
    scaler, clf = pipeline.named_steps["standardscaler"], pipeline.named_steps["logisticregression"]

    proba = pipeline.predict_proba(X_test)[:, 1]
    bands = np.array([band(p) for p in proba])
    band_stats = {
        name: {
            "count": int((bands == name).sum()),
            "diabetic_rate": float(y_test[bands == name].mean()) if (bands == name).any() else None,
        }
        for name in ("Low", "Medium", "High")
    }

    data_hash = data_fingerprint(get_settings().data_path)
    trained_at = datetime.now(UTC)
    params = {
        "features": FEATURE_COLUMNS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "bands": BANDS,
    }
    params_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    card = {
        "model_version": f"logreg-{params_hash}",
        "trained_at": trained_at.isoformat(),
        "algorithm": "StandardScaler + LogisticRegression",
        "sklearn_version": sklearn.__version__,
        "data_sha256_prefix": data_hash,
        "seed": args.seed,
        "features": FEATURE_COLUMNS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "bands": BANDS,
        "metrics": {
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "test_prevalence": float(y_test.mean()),
            "cv_auc_mean": float(cv_auc.mean()),
            "cv_auc_std": float(cv_auc.std()),
            "test_auc": float(roc_auc_score(y_test, proba)),
            "test_brier": float(brier_score_loss(y_test, proba)),
            "test_accuracy_at_0_5": float(accuracy_score(y_test, proba >= 0.5)),
            "test_bands": band_stats,
        },
        "test_patient_ids": sorted(ids_test.tolist(), key=int),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(card, indent=2))

    m = card["metrics"]
    print(f"Saved {card['model_version']} to {args.out}")
    print(f"  CV AUC {m['cv_auc_mean']:.3f} ± {m['cv_auc_std']:.3f} | test AUC {m['test_auc']:.3f} | Brier {m['test_brier']:.3f}")
    for name, stats in band_stats.items():
        rate = "n/a" if stats["diabetic_rate"] is None else f"{stats['diabetic_rate']:.0%}"
        print(f"  {name:<6} n={stats['count']:<4} diabetic rate {rate}")
    print("  coefficients (per 1 SD):")
    for feature, coef in sorted(zip(FEATURE_COLUMNS, card["coefficients"]), key=lambda x: -abs(x[1])):
        print(f"    {feature:<26} {coef:+.3f}")


if __name__ == "__main__":
    main()
