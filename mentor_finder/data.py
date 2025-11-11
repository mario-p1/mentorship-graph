from pathlib import Path

import pandas as pd


def load_raw_committee_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [
        "thesis_title_mk",
        "student",
        "mentor",
        "c1",
        "c2",
        "thesis_application_date",
        "thesis_status",
        "graduation_thesis_desc_mk",
        "thesis_desc_en",
        "thesis_title_en",
    ]
    return df
