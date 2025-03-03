import sys
from typing import Optional

import pandas as pd

from definitions import DATE_FORMAT
from src.utils.data_loaders import load_ts_data


def parse_nrel118_loads_ts(
    raw_data: str, path_parsed_data: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Parse raw time-series data of loads from the NREL-118 dataset.

    Args:
        raw_data: Path to the raw data.
        path_parsed_data: Path to save parsed data.

    Returns:
        Parsed data or None if `path_parsed_data` is passed and the data were saved.
    """
    # Load data
    name_pattern = r"Load(?P<name>\w+)RT\.csv"
    load_ts = load_ts_data(folder_path=raw_data, name_pattern=name_pattern)

    # Change column names
    load_ts.rename(
        columns={"name": "region_name", "value": "region_load"}, inplace=True
    )
    load_ts["region_name"] = load_ts["region_name"].str.lower()

    # Unify date format
    load_ts["datetime"] = load_ts["datetime"].dt.strftime(DATE_FORMAT)

    # Return results
    load_ts.sort_values(["datetime", "region_name"], inplace=True, ignore_index=True)
    if path_parsed_data:
        load_ts.to_csv(path_parsed_data, header=True, index=False)
    else:
        return load_ts


if __name__ == "__main__":
    # Check params
    if len(sys.argv) != 3:
        raise ValueError(
            "Incorrect arguments. Usage:\n\tpython "
            "parse_nrel118_loads_ts.py path_raw_nrel118_loads_ts path_parsed_data\n"
        )

    # Run
    parse_nrel118_loads_ts(raw_data=sys.argv[1], path_parsed_data=sys.argv[2])
