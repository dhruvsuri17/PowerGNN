import sys
import os

# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from definitions import S_BASE_MVA, F_HZ
from src.data import (
    parse_nrel118_buses, parse_nrel118_lines, prepare_buses, prepare_branches,
    parse_jeas118_buses, parse_jeas118_lines, parse_jeas118_loads, parse_jeas118_trafos,
    parse_nrel118_escalators_ts, parse_nrel118_gens, parse_nrel118_hydros_nondisp_ts,
    parse_nrel118_hydros_ts, parse_nrel118_loads_ts, parse_nrel118_outages_ts,
    parse_nrel118_solars_ts, parse_nrel118_winds_ts, prepare_gens, prepare_gens_ts,
    prepare_loads, prepare_loads_ts, transform_gens_escalated_ts, transform_loads,
    transform_outages_ts, transform_gens, transform_gens_ts
)
from src.power_flow.builders import PandaPowerFlowBuilder
import pandapower.plotting.plotly as pplotly


def run_power_flow():
    """Loads data, processes it, and returns a PandaPowerFlowBuilder instance."""

    # Define paths
    PATH_NREL118 = os.path.join(project_root, "data", "raw", "nrel118")
    PATH_JEAS118 = os.path.join(project_root, "data", "raw", "jeas118")
    PATH_MANUAL = os.path.join(project_root, "data", "raw", "manual")

    # Load buses
    path_nrel118_buses = os.path.join(PATH_NREL118, "additional-files-mti-118", "Buses.csv")
    nrel118_buses = parse_nrel118_buses(raw_data=path_nrel118_buses)

    path_jeas_118_buses = os.path.join(PATH_JEAS118, "JEAS_IEEE118.docx")
    jeas118_buses = parse_jeas118_buses(path_jeas_118_buses)

    path_bus_coordinates = os.path.join(PATH_MANUAL, "bus_coordinates.csv")
    buses = prepare_buses(
        parsed_nrel118_buses=nrel118_buses,
        parsed_jeas118_buses=jeas118_buses,
        bus_coordinates=path_bus_coordinates,
    )

    path_nrel118_lines = os.path.join(PATH_NREL118, "additional-files-mti-118", "Lines.csv")
    nrel118_lines = parse_nrel118_lines(raw_data=path_nrel118_lines)

    path_jeas118_lines = os.path.join(PATH_JEAS118, "JEAS_IEEE118.docx")
    jeas118_lines = parse_jeas118_lines(raw_data=path_jeas118_lines)

    path_jeas118_trafos = os.path.join(PATH_JEAS118, "JEAS_IEEE118.docx")
    jeas118_trafos = parse_jeas118_trafos(raw_data=path_jeas118_trafos)

    branches = prepare_branches(
        parsed_nrel118_lines=nrel118_lines,
        parsed_jeas118_lines=jeas118_lines,
        parsed_jeas118_trafos=jeas118_trafos,
        prepared_buses=buses,
    )

    path_nrel118_loads_ts = os.path.join(PATH_NREL118, "Input files", "RT", "Load")
    nrel118_loads_ts = parse_nrel118_loads_ts(raw_data=path_nrel118_loads_ts)

    path_jeas118_loads = os.path.join(PATH_JEAS118, "JEAS_IEEE118.docx")
    jeas118_loads = parse_jeas118_loads(raw_data=path_jeas118_loads)

    transformed_loads = transform_loads(
        parsed_nrel118_buses=nrel118_buses, parsed_jeas118_loads=jeas118_loads
    )

    loads = prepare_loads(transformed_loads=transformed_loads)

    loads_ts = prepare_loads_ts(
        transformed_loads=transformed_loads, parsed_nrel118_loads_ts=nrel118_loads_ts
    )

    path_nrel118_gens = os.path.join(
        PATH_NREL118, "additional-files-mti-118", "Generators.csv"
    )
    nrel118_gens = parse_nrel118_gens(raw_data=path_nrel118_gens)

    # Hydro gens
    path_nrel118_hydros_ts = os.path.join(PATH_NREL118, "Input files", "Hydro")
    nrel118_hydros_ts = parse_nrel118_hydros_ts(raw_data=path_nrel118_hydros_ts)

    # Solar gens
    path_nrel118_solars_ts = os.path.join(PATH_NREL118, "Input files", "RT", "Solar")
    nrel118_solars_ts = parse_nrel118_solars_ts(raw_data=path_nrel118_solars_ts)

    # Wind gens
    path_nrel118_winds_ts = os.path.join(PATH_NREL118, "Input files", "RT", "Wind")
    nrel118_winds_ts = parse_nrel118_winds_ts(raw_data=path_nrel118_winds_ts)

    # Non-dispatchable hydro gens
    path_nrel118_hydros_nondisp_ts = os.path.join(
        PATH_NREL118,
        "additional-files-mti-118",
        "Hydro_nondipatchable.csv",
    )
    nrel118_hydros_nondisp_ts = parse_nrel118_hydros_nondisp_ts(
        raw_data=path_nrel118_hydros_nondisp_ts
    )

    # Escalators data
    path_nrel118_escalators_ts = os.path.join(
        PATH_NREL118, "additional-files-mti-118", "Escalators.csv"
    )
    nrel118_escalators_ts = parse_nrel118_escalators_ts(raw_data=path_nrel118_escalators_ts)

    # Outages
    path_nrel118_outages_ts = os.path.join(
        PATH_NREL118, "Input files", "Others", "GenOut.csv"
    )
    nrel118_outages_ts = parse_nrel118_outages_ts(raw_data=path_nrel118_outages_ts)

    transformed_gens = transform_gens(
        parsed_nrel118_gens=nrel118_gens,
        prepared_buses=buses,
    )
    transformed_outages_ts = transform_outages_ts(
        parsed_nrel118_outages_ts=nrel118_outages_ts
    )
    transformed_gens_escalated_ts = transform_gens_escalated_ts(
        transformed_gens=transformed_gens,
        parsed_nrel118_escalators_ts=nrel118_escalators_ts,
    )
    transformed_gens_ts = transform_gens_ts(
        transformed_gens=transformed_gens,
        parsed_nrel118_winds_ts=nrel118_winds_ts,
        parsed_nrel118_solars_ts=nrel118_solars_ts,
        parsed_nrel118_hydros_ts=nrel118_hydros_ts,
        parsed_nrel118_hydros_nondisp_ts=nrel118_hydros_nondisp_ts,
    )

    gens_ts = prepare_gens_ts(
        transformed_gens=transformed_gens,
        transformed_gens_ts=transformed_gens_ts,
        transformed_outages_ts=transformed_outages_ts,
        transformed_gens_escalated_ts=transformed_gens_escalated_ts,
    )

    gens = prepare_gens(transformed_gens=transformed_gens, prepared_gens_ts=gens_ts)

    # Create the builder
    builder = PandaPowerFlowBuilder(f_hz=F_HZ, s_base_mva=S_BASE_MVA)

    # Load data
    builder.load_data(
        buses=buses,
        branches=branches,
        loads=loads,
        loads_ts=loads_ts,
        gens=gens,
        gens_ts=gens_ts,
    )

    print("Power flow model successfully built.")
    return builder


if __name__ == "__main__":
    builder = run_power_flow()
    print("Finished running power flow simulation.")
