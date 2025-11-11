from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from mentor_finder.embedding import embed_text


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


def build_graph(df: pd.DataFrame) -> tuple[HeteroData, dict[str, Any]]:
    # Build thesis features
    desc_embeddings = embed_text(df["thesis_desc_en"].tolist())
    thesis_features = torch.from_numpy(desc_embeddings)

    # Mentors
    mentors = sorted(df["mentor"].unique().tolist())
    mentors_dict = {mentor: index for index, mentor in enumerate(mentors)}

    mentor_features = torch.ones((len(mentors), 1))

    # Supervises relation
    supervises_source = df["mentor"].apply(lambda mentor: mentors_dict[mentor])
    supervises_destination = df.index.tolist()
    supervises_features = torch.vstack(
        [torch.IntTensor(supervises_source), torch.IntTensor(supervises_destination)]
    )

    # Build graph
    graph = HeteroData()
    graph["thesis"].x = thesis_features
    graph["mentor"].x = mentor_features
    graph["mentor", "supervises", "thesis"].edge_index = supervises_features

    return graph, {"mentors_dict": mentors_dict}
