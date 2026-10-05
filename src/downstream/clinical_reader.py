"""Time-restricted UKB clinical covariate readers from the original experiment.

No legacy cohort construction, unrelated disease scores, or evaluation driver.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv


FIELD_IDS = (
    "31", "50", "53", "93", "94", "1239", "2443", "4079", "4080",
    "6153", "6177", "20002", "20116", "21000", "21002", "21003",
    "42000", "42006",
)


SELF_REPORTED = {
    "hypertension": {1065, 1072},
    "diabetes": {1220, 1222, 1223, 1521},
    "heart_failure": {1076},
    "myocardial_infarction": {1075},
    "stroke_tia": {1081, 1082, 1086, 1491, 1583},
    "peripheral_vascular": {1067, 1087},
}


def csv_header(path: Path) -> list[str]:
    csv.field_size_limit(1 << 30)
    with path.open(newline="") as handle:
        return next(csv.reader(handle))


def columns_for(field_id: str, names: list[str]) -> list[str]:
    return [name for name in names
            if name == field_id or name.startswith(field_id + "-")]


def instance(column: str) -> int:
    return int(column.split("-", 1)[1].split(".", 1)[0])


def read_target_rows(raw_csv: Path, target_eids: np.ndarray) -> pd.DataFrame:
    names = csv_header(raw_csv)
    grouped = {field: columns_for(field, names) for field in FIELD_IDS}
    missing = [field for field, columns in grouped.items() if not columns]
    if missing:
        raise RuntimeError(f"UKB export lacks fields: {missing}")
    include = ["eid", *sorted({c for columns in grouped.values() for c in columns})]
    date_fields = {"53", "42000", "42006"}
    types: dict[str, pa.DataType] = {"eid": pa.int64()}
    for column in include[1:]:
        types[column] = pa.string() if column.split("-", 1)[0] in date_fields else pa.float64()
    reader = pacsv.open_csv(
        raw_csv,
        read_options=pacsv.ReadOptions(use_threads=True, block_size=64 << 20),
        convert_options=pacsv.ConvertOptions(
            include_columns=include,
            column_types=types,
            strings_can_be_null=True,
            null_values=[""],
        ),
    )
    wanted = pa.array(target_eids.astype(np.int64))
    retained: list[pa.Table] = []
    for batch in reader:
        mask = pc.is_in(batch.column("eid"), value_set=wanted)
        filtered = pa.Table.from_batches([batch]).filter(mask)
        if filtered.num_rows:
            retained.append(filtered)
    if not retained:
        raise RuntimeError("no target EIDs found in raw UKB export")
    frame = pa.concat_tables(retained).to_pandas()
    frame["eid"] = frame["eid"].astype("Int64").astype(str)
    if frame["eid"].duplicated().any():
        raise RuntimeError("raw UKB export contains duplicate EIDs")
    expected = set(target_eids.astype(str))
    observed = set(frame["eid"])
    if expected != observed:
        raise RuntimeError(f"raw export coverage mismatch: missing={len(expected-observed)}")
    return frame.set_index("eid").loc[target_eids.astype(str)].reset_index()


def assessment_dates(raw: pd.DataFrame) -> np.ndarray:
    dates = np.full((len(raw), 4), np.datetime64("NaT"), dtype="datetime64[D]")
    for column in [c for c in raw if c.startswith("53-")]:
        dates[:, instance(column)] = pd.to_datetime(
            raw[column], errors="coerce").to_numpy(dtype="datetime64[D]")
    return dates


def numeric_by_visit(raw: pd.DataFrame, prefixes: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((len(raw), 4), np.nan, np.float64)
    observed = np.zeros((len(raw), 4), bool)
    for visit in range(4):
        columns = [c for c in raw if c.split("-", 1)[0] in prefixes and instance(c) == visit]
        if not columns:
            continue
        block = raw[columns].apply(pd.to_numeric, errors="coerce")
        block = block.mask(block.isin([-1, -3]))
        observed[:, visit] = block.notna().any(axis=1).to_numpy()
        values[:, visit] = block.mean(axis=1, skipna=True).to_numpy()
    return values, observed


def latest_prior(values: np.ndarray, observed: np.ndarray,
                 visits: np.ndarray, retina: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    allowed = observed & ~np.isnat(visits) & (visits <= retina[:, None])
    ordinal = visits.astype("datetime64[D]").astype(np.int64)
    ordinal[~allowed] = np.iinfo(np.int64).min
    best = ordinal.argmax(axis=1)
    known = allowed.any(axis=1)
    out = values[np.arange(len(values)), best].astype(np.float64)
    out[~known] = np.nan
    return out, known


def categorical_presence_by_visit(raw: pd.DataFrame, prefixes: tuple[str, ...],
                                  positive_codes: set[int]) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(raw), 4), np.float64)
    observed = np.zeros((len(raw), 4), bool)
    for visit in range(4):
        columns = [c for c in raw if c.split("-", 1)[0] in prefixes and instance(c) == visit]
        if not columns:
            continue
        block = raw[columns].apply(pd.to_numeric, errors="coerce")
        observed[:, visit] = block.notna().any(axis=1).to_numpy()
        values[:, visit] = block.isin(positive_codes).any(axis=1).astype(float).to_numpy()
    return values, observed


def any_prior_condition(raw: pd.DataFrame, visits: np.ndarray, retina: np.ndarray,
                        codes: set[int]) -> tuple[np.ndarray, np.ndarray]:
    presence, observed = categorical_presence_by_visit(raw, ("20002",), codes)
    allowed = observed & ~np.isnat(visits) & (visits <= retina[:, None])
    return (presence.astype(bool) & allowed).any(axis=1), allowed.any(axis=1)
