import sys
from typing import Optional

import pandas as pd

from definitions import DATE_FORMAT, GEN_TYPES
from src.utils.data_loaders import load_df_data


def parse_nrel118_hydros_nondisp_ts(
    raw_data: str | pd.DataFrame, path_parsed_data: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Parse raw data of non-dispatchable hydro plants from the NREL-118 dataset.

    Args:
        raw_data: Path or dataframe with raw data.
        path_parsed_data: Path to save parsed data.

    Returns:
        Parsed data or None if `path_parsed_data` is passed and the data were saved.
    """
    dtypes = {"Generator": str, "Value": float, "Timeslice": str}
    hydros = load_df_data(data=raw_data, dtypes=dtypes)

    # Rename variables
    hydros.rename(
        columns={
            "Generator": "gen_name",
            "Value": "p_mw",
            "Timeslice": "month",
        },
        inplace=True,
    )

    # Drop unused variables
    hydros = hydros[~hydros["month"].isna()].reset_index(drop=True)

    # Unify generator names
    name_pattern = r"^(?P<gen_type>[\w\s]+)\s(?P<gen_number>\d+)$"
    names = hydros["gen_name"].str.extract(pat=name_pattern, expand=True)
    names["gen_type"].replace(GEN_TYPES, inplace=True)
    hydros["gen_name"] = names["gen_type"] + "_" + names["gen_number"].str.zfill(3)

    # Convert datetime
    hydros["year"] = 2024
    hydros["day"] = 1
    hydros["month"] = hydros["month"].str.lstrip("M").astype(int)
    hydros["datetime"] = pd.to_datetime(hydros[["year", "month", "day"]]).dt.strftime(
        DATE_FORMAT
    )

    # Return results
    hydros.sort_values(["datetime", "gen_name"], inplace=True, ignore_index=True)
    cols = ["datetime", "gen_name", "p_mw"]
    if path_parsed_data:
        hydros[cols].to_csv(path_parsed_data, header=True, index=False)
    else:
        return hydros[cols]


if __name__ == "__main__":
    # Check params
    if len(sys.argv) != 3:
        raise ValueError(
            "Incorrect arguments. Usage:\n\tpython "
            "parse_nrel118_hydros_nondisp_ts.py path_raw_nrel118_hydros_nondisp "
            "path_parsed_data\n"
        )

    # Run
    parse_nrel118_hydros_nondisp_ts(raw_data=sys.argv[1], path_parsed_data=sys.argv[2])
