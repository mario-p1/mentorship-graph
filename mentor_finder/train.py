from pathlib import Path


import pandas as pd

from mentor_finder.data import build_graph


def main():
    pd.options.display.max_rows = 20
    pd.options.display.max_columns = 20

    df = pd.read_csv(
        Path(__file__).parent.parent / "data" / "committee_train.csv",
    )
    df = df.head(5)

    data, metadata = build_graph(df)

    print(data)
    print(metadata)


if __name__ == "__main__":
    main()
