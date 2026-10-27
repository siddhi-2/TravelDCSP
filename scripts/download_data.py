"""
Download the REAL TravelPlanner data (run this on YOUR machine -- needs internet):

    pip install datasets gdown pandas
    python scripts/download_data.py

Creates:
    data/travelplanner/train.csv         (45 queries)
    data/travelplanner/validation.csv    (180 queries)    -> 45 + 180 = 225 queries with reference plans
    data/travelplanner/database/...      (the six tool databases: flights, distance matrix,
                                          accommodations, restaurants, attractions, city lists)

The 1,000-query 'test' split has no reference answers (leaderboard only), so it is not used.

If gdown is blocked, download database.zip manually from
  https://drive.google.com/file/d/1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE/view
and unzip it so that data/travelplanner/database/ exists.
"""
import os
import sys
import zipfile

OUT = os.path.join("data", "travelplanner")
DRIVE_ID = "1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE"


def get_queries():
    from datasets import load_dataset
    os.makedirs(OUT, exist_ok=True)
    for split in ("train", "validation"):
        ds = load_dataset("osunlp/TravelPlanner", split)[split]
        df = ds.to_pandas()
        path = os.path.join(OUT, f"{split}.csv")
        df.to_csv(path, index=False)
        print(f"  {split}: {len(df)} queries -> {path}   columns: {list(df.columns)}")


def get_database():
    import gdown
    os.makedirs(OUT, exist_ok=True)
    zpath = os.path.join(OUT, "database.zip")
    if not os.path.isdir(os.path.join(OUT, "database")):
        gdown.download(id=DRIVE_ID, output=zpath, quiet=False)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(OUT)
    print("  database folder:", os.path.join(OUT, "database"))


if __name__ == "__main__":
    print("1/2 queries (Hugging Face: osunlp/TravelPlanner)")
    try:
        get_queries()
    except Exception as e:                                       # noqa: BLE001
        sys.exit(f"Query download failed: {e}\nTip: `huggingface-cli login` may be required.")
    print("2/2 database (Google Drive)")
    try:
        get_database()
    except Exception as e:                                       # noqa: BLE001
        sys.exit(f"Database download failed: {e}\nDownload database.zip manually (see top of this file).")
    print("\nDone. Run:\n  python run_experiment.py --data_dir data/travelplanner "
          "--db data/travelplanner/database --split all")
