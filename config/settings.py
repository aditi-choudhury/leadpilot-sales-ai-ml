from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "leads.csv"
MODEL = ROOT / "models" / "pipeline.joblib"
METRICS = ROOT / "models" / "metrics.json"
DATABASE = ROOT / "data" / "sales.db"
SEED = 42
HIGH = 0.70
MEDIUM = 0.40
