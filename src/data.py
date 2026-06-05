"""
Reusable data loading and cleaning for the Czech Bank credit card affinity model.

Extracts shared logic from 00_eda.ipynb so that downstream notebooks can
`from src.data import load_raw_data, load_study_table` without re-implementing.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ── Shared Constants ─────────────────────────────────────────────────────────

RANDOM_STATE = 42
ROLLUP_MONTHS = 12
LAG_MONTHS = 1
TOTAL_HISTORY_MONTHS = ROLLUP_MONTHS + LAG_MONTHS  # 13
JUNIOR_AGE_CUTOFF = 21

TRAIN_RATIO = 0.8
TEST_RATIO = 0.2

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATH_RAW = PROJECT_ROOT / "data" / "raw"
PATH_PROCESSED = PROJECT_ROOT / "data" / "processed"
PATH_STUDY_TABLE = PATH_PROCESSED / "study_table.parquet"
PATH_FEATURE_MATRIX = PATH_PROCESSED / "feature_matrix.parquet"

# Plotting defaults
FIG_SIZE_DEFAULT = (14, 6)
FIG_SIZE_LARGE = (16, 12)


# ── Helper Functions ─────────────────────────────────────────────────────────


def parse_date(col: pd.Series) -> pd.Series:
    """Convert YYMMDD (int/str) column to datetime. Idempotent."""
    if pd.api.types.is_datetime64_any_dtype(col):
        return col
    col_str = col.astype(str).str[:6].str.zfill(6)
    return pd.to_datetime("19" + col_str, format="%Y%m%d", errors="coerce")


def extract_gender_and_birth(birth_number: int) -> pd.Series:
    """Decode client birth_number into (gender, birth_date)."""
    birth_str = str(birth_number).zfill(6)
    year = int(birth_str[:2]) + 1900
    month = int(birth_str[2:4])
    day = int(birth_str[4:])

    if month > 50:
        gender = "F"
        month -= 50
    else:
        gender = "M"

    return pd.Series(
        [gender, pd.to_datetime(f"{year}-{month:02d}-{day:02d}", errors="coerce")]
    )


# ── Czech String Mappings ────────────────────────────────────────────────────

FREQ_MAPPING = {
    "POPLATEK MESICNE": "monthly",
    "POPLATEK TYDNE": "weekly",
    "POPLATEK PO OBRATU": "after transaction",
}

TRANS_TYPE_MAPPING = {"PRIJEM": "credit", "VYDAJ": "withdrawal"}

TRANS_OP_MAPPING = {
    "VYBER KARTOU": "credit_card_withdrawal",
    "VKLAD": "credit_in_cash",
    "PREVOD Z UCTU": "collection_from_another_bank",
    "VYBER": "withdrawal_in_cash",
    "PREVOD NA UCET": "remittance_to_another_bank",
}

K_SYMBOL_MAPPING = {
    "POJISTNE": "insurance_payment",
    "SIPO": "household_payment",
    "LEASING": "leasing",
    "UVER": "loan_payment",
    "SLUZBY": "payment_for_statement",
    "UROK": "interest_credited",
    "SANKC. UROK": "sanction_interest",
    "DUCHOD": "old_age_pension",
}


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_raw_data(path_raw: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Load all 8 raw CSVs, apply cleaning:
      - Rename district A1 -> district_id
      - Decode Czech strings (frequency, type, operation, k_symbol)
      - Parse dates (account, card, loan, trans)
      - Extract gender and birth_date from client.birth_number

    Returns dict with keys: account, card, client, disp, district, loan, order, trans
    """
    if path_raw is None:
        path_raw = PATH_RAW
    path_raw = Path(path_raw)

    # Load CSVs
    account = pd.read_csv(path_raw / "account.csv", sep=";")
    card = pd.read_csv(path_raw / "card.csv", sep=";")
    client = pd.read_csv(path_raw / "client.csv", sep=";")
    disp = pd.read_csv(path_raw / "disp.csv", sep=";")
    district = pd.read_csv(path_raw / "district.csv", sep=";")
    loan = pd.read_csv(path_raw / "loan.csv", sep=";")
    order = pd.read_csv(path_raw / "order.csv", sep=";")
    trans = pd.read_csv(path_raw / "trans.csv", sep=";", low_memory=False)

    # Rename district PK
    district.rename(columns={"A1": "district_id"}, inplace=True)

    # Decode Czech strings
    account["frequency"] = account["frequency"].map(FREQ_MAPPING)
    trans["type"] = trans["type"].map(TRANS_TYPE_MAPPING)
    trans["operation"] = (
        trans["operation"].map(TRANS_OP_MAPPING).fillna(trans["operation"])
    )
    trans["k_symbol"] = (
        trans["k_symbol"].map(K_SYMBOL_MAPPING).fillna(trans["k_symbol"])
    )
    order["k_symbol"] = (
        order["k_symbol"].map(K_SYMBOL_MAPPING).fillna(order["k_symbol"])
    )

    # Parse dates
    account["date"] = parse_date(account["date"])
    card["issued"] = parse_date(card["issued"])
    loan["date"] = parse_date(loan["date"])
    trans["date"] = parse_date(trans["date"])

    # Extract gender & birth_date
    client[["gender", "birth_date"]] = client["birth_number"].apply(
        extract_gender_and_birth
    )
    client = client.drop(columns=["birth_number"])

    return {
        "account": account,
        "card": card,
        "client": client,
        "disp": disp,
        "district": district,
        "loan": loan,
        "order": order,
        "trans": trans,
    }


def load_study_table(path: str | Path | None = None) -> pd.DataFrame:
    """Load study_table from parquet."""
    if path is None:
        path = PATH_STUDY_TABLE
    df = pd.read_parquet(path)
    for col in ["event_date", "rollup_start", "rollup_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def assign_pseudo_event_dates(
    account_df: pd.DataFrame,
    eligible_non_card_ids: set[int],
    empirical_months: pd.Series,
    max_data_date: pd.Timestamp,
    first_trans: pd.Series,
    rollup_months: int = ROLLUP_MONTHS,
    lag_months: int = LAG_MONTHS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Weist jedem eligible Nicht-Karten-Konto ein Pseudo-Ereignisdatum zu,
    basierend auf der empirischen Verteilung aus dem Trainingsset.

    Accounts ohne feasible Bereich werden ausgeschlossen (unzureichende Historie).
    Rollup-Fenster: [event - (rollup_months + lag_months) ... event - lag_months]
    """
    rng = np.random.default_rng(random_state)
    total_history = rollup_months + lag_months

    # Nur eligible Nicht-Karten-Konten
    non_card = account_df[account_df["account_id"].isin(eligible_non_card_ids)].copy()
    non_card = non_card.rename(columns={"date": "account_creation_date"})
    non_card = non_card.merge(first_trans, on="account_id", how="left")

    empirical_values = empirical_months.values
    max_period = max_data_date.to_period("M")

    # Vectorize: Berechne min/max months pro Konto via Period-Arithmetik
    creation_periods = non_card["account_creation_date"].dt.to_period("M")
    first_tx_periods = non_card["first_trans_date"].dt.to_period("M")

    # min_months: event >= first_tx + total_history Monate
    non_card["min_months"] = (
        (first_tx_periods + total_history).astype("int64")
        - creation_periods.astype("int64")
    ).clip(lower=total_history)

    # max_months: event <= max_data_date
    non_card["max_months"] = max_period.ordinal - creation_periods.astype("int64")

    # Ausschluss: Konten ohne machbaren Bereich (zu spät eröffnet)
    infeasible_mask = non_card["max_months"] < non_card["min_months"]
    n_infeasible = infeasible_mask.sum()
    if n_infeasible > 0:
        print(f"  Ausgeschlossen (unzureichende Historie): {n_infeasible} Konten")
    non_card = non_card[~infeasible_mask].copy()

    # Per-row sampling (unvermeidbar wegen row-spezifischer feasible ranges)
    def _sample_row(row):
        feasible = empirical_values[
            (empirical_values >= row["min_months"])
            & (empirical_values <= row["max_months"])
        ]
        if len(feasible) > 0:
            return int(rng.choice(feasible))
        # Fallback: Mitte des machbaren Bereichs (sollte nie eintreten)
        return int((row["min_months"] + row["max_months"]) // 2)

    non_card["months_to_event"] = non_card.apply(_sample_row, axis=1)

    # Event-Datum und Rollup-Fenster berechnen
    non_card["event_date"] = non_card.apply(
        lambda r: r["account_creation_date"]
        + pd.DateOffset(months=int(r["months_to_event"])),
        axis=1,
    )
    non_card["rollup_end"] = non_card["event_date"].apply(
        lambda d: (d.to_period("M") - lag_months).to_timestamp()
    )
    non_card["rollup_start"] = non_card["event_date"].apply(
        lambda d: (d.to_period("M") - total_history).to_timestamp()
    )
    non_card["target"] = 0

    return non_card[
        [
            "account_id",
            "account_creation_date",
            "event_date",
            "rollup_start",
            "rollup_end",
            "months_to_event",
            "target",
        ]
    ]


def set_plot_style():
    """Apply the project's standard plotting style."""
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid", context="paper", palette="viridis")
    plt.rcParams["figure.figsize"] = FIG_SIZE_DEFAULT
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
