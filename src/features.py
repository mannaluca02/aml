"""
Feature engineering for the credit card affinity model.

Provides:
  - Transaction rollup filtering
  - tsfresh-based automatic feature extraction
  - Hand-crafted baseline features (5 required by assignment)
  - Static/contextual features (district, loan, order, account)
  - Master builder that assembles the full feature matrix
  - Feature overview table for the deliverable
"""

import warnings

import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

# ── Feature Registries ───────────────────────────────────────────────────────

BASELINE_FEATURE_COLS = [
    "age_at_event",
    "gender",
    "district_avg_salary",
    "balance_last",
    "turnover",
]

TSFRESH_FEATURE_PREFIX = "tsf_"

STATIC_FEATURE_COLS = [
    # Client
    "age_at_event",
    "gender",
    # District
    "district_avg_salary",
    "district_unemployment_96",
    "district_urban_ratio",
    "district_inhabitants",
    "district_entrepreneurs",
    "district_crimes_96",
    # Loan
    "has_loan",
    "loan_amount",
    "loan_monthly_payment",
    # Order
    "order_count",
    "order_total_amount",
    "has_insurance_order",
    "has_household_order",
    "has_loan_order",
    # Account
    "freq_monthly",
    "freq_weekly",
    "freq_after_transaction",
    "months_to_event",
]


# ── Transaction Rollup Filter ───────────────────────────────────────────────


def filter_transactions_to_rollup(
    trans: pd.DataFrame,
    study_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only transactions within each account's [rollup_start, rollup_end] window.

    Parameters
    ----------
    trans : DataFrame with columns account_id, date, amount, balance, ...
    study_table : DataFrame with columns account_id, rollup_start, rollup_end

    Returns
    -------
    Filtered transactions DataFrame (original columns only, no study_table cols).
    """
    merged = trans.merge(
        study_table[["account_id", "rollup_start", "rollup_end"]],
        on="account_id",
        how="inner",
    )
    mask = (merged["date"] >= merged["rollup_start"]) & (
        merged["date"] <= merged["rollup_end"]
    )
    filtered = merged.loc[mask].drop(columns=["rollup_start", "rollup_end"])
    return filtered


# ── tsfresh Feature Extraction ───────────────────────────────────────────────


def compute_tsfresh_features(
    trans_rollup: pd.DataFrame,
    study_table: pd.DataFrame,
    fc_parameters: dict | None = None,
) -> pd.DataFrame:
    """
    Extract automatic features from transaction time series using tsfresh.

    Extracts features from `amount` and `balance` columns, sorted by date.
    Prefixes all resulting columns with 'tsf_'.

    Parameters
    ----------
    trans_rollup : Rollup-filtered transactions (output of filter_transactions_to_rollup)
    study_table : Study table (used only to ensure all accounts are represented)
    fc_parameters : tsfresh feature calculator parameters. Defaults to EfficientFCParameters.

    Returns
    -------
    DataFrame indexed by account_id with ~1'554 tsfresh features (~777 per time series).
    """
    if fc_parameters is None:
        fc_parameters = EfficientFCParameters()

    # Prepare tsfresh input: account_id, date (sort), amount, balance
    ts_input = (
        trans_rollup[["account_id", "date", "amount", "balance"]]
        .sort_values(["account_id", "date"])
        .copy()
    )

    # tsfresh needs a numeric sort column
    ts_input["time_idx"] = ts_input.groupby("account_id").cumcount()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = extract_features(
            ts_input[["account_id", "time_idx", "amount", "balance"]],
            column_id="account_id",
            column_sort="time_idx",
            default_fc_parameters=fc_parameters,
            n_jobs=0,
            disable_progressbar=False,
        )
        # Impute NaN/inf values (inside warning block — tsfresh warns about
        # ~242 high-frequency FFT columns with no finite values, which is expected)
        impute(features)

    # Prefix column names
    features.columns = [f"{TSFRESH_FEATURE_PREFIX}{c}" for c in features.columns]

    # Ensure index is named account_id
    features.index.name = "account_id"

    return features


# ── Baseline Features (5 required) ──────────────────────────────────────────


def compute_baseline_features(
    trans_rollup: pd.DataFrame,
    study_table: pd.DataFrame,
    client: pd.DataFrame,
    disp: pd.DataFrame,
    account: pd.DataFrame,
    district: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the 5 baseline features required by the assignment:
      1. balance_last  — last balance in rollup window (wealth/Vermögen)
      2. turnover      — sum of abs(amount) in rollup window (Umsatz)
      3. age_at_event  — age in years at event_date
      4. gender        — binary (1=M, 0=F)
      5. district_avg_salary — A11 via account -> district
    """
    # --- Transaction-based features ---
    # Last balance per account in rollup window
    sorted_trans = trans_rollup.sort_values(["account_id", "date"])
    balance_last = (
        sorted_trans.groupby("account_id")["balance"].last().rename("balance_last")
    )

    # Turnover = sum of absolute amounts
    turnover = (
        sorted_trans.groupby("account_id")["amount"]
        .apply(lambda x: x.abs().sum())
        .rename("turnover")
    )

    tx_features = pd.concat([balance_last, turnover], axis=1).reset_index()

    # --- Client-based features (via OWNER disposition) ---
    owners = disp[disp["type"] == "OWNER"][["account_id", "client_id"]]
    client_info = owners.merge(
        client[["client_id", "gender", "birth_date"]], on="client_id", how="left"
    )
    # Join with study_table for event_date
    client_info = client_info.merge(
        study_table[["account_id", "event_date"]], on="account_id", how="left"
    )
    client_info["age_at_event"] = (
        client_info["event_date"] - client_info["birth_date"]
    ).dt.days / 365.25
    client_info["gender"] = (client_info["gender"] == "M").astype(int)

    # --- District-based features ---
    district_salary = account[["account_id", "district_id"]].merge(
        district[["district_id", "A11"]].rename(columns={"A11": "district_avg_salary"}),
        on="district_id",
        how="left",
    )

    # --- Combine ---
    result = study_table[["account_id"]].copy()
    result = result.merge(tx_features, on="account_id", how="left")
    result = result.merge(
        client_info[["account_id", "age_at_event", "gender"]],
        on="account_id",
        how="left",
    )
    result = result.merge(
        district_salary[["account_id", "district_avg_salary"]],
        on="account_id",
        how="left",
    )

    return result


# ── Static / Contextual Features ────────────────────────────────────────────


def compute_static_features(
    study_table: pd.DataFrame,
    client: pd.DataFrame,
    disp: pd.DataFrame,
    account: pd.DataFrame,
    district: pd.DataFrame,
    loan: pd.DataFrame,
    order: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute static/contextual features per account:
      - Client: age_at_event, gender (reused from baseline)
      - District: avg_salary, unemployment, urban_ratio, inhabitants, entrepreneurs, crimes
      - Loan: has_loan, loan_amount, loan_monthly_payment (temporal guard: loan.date <= rollup_end)
      - Order: order_count, order_total_amount, has_insurance/household/loan_order
      - Account: frequency one-hot, months_to_event
    """
    result = study_table[["account_id"]].copy()

    # --- Client features (via OWNER) ---
    owners = disp[disp["type"] == "OWNER"][["account_id", "client_id"]]
    client_info = owners.merge(
        client[["client_id", "gender", "birth_date"]], on="client_id", how="left"
    )
    client_info = client_info.merge(
        study_table[["account_id", "event_date"]], on="account_id", how="left"
    )
    client_info["age_at_event"] = (
        client_info["event_date"] - client_info["birth_date"]
    ).dt.days / 365.25
    client_info["gender"] = (client_info["gender"] == "M").astype(int)
    result = result.merge(
        client_info[["account_id", "age_at_event", "gender"]],
        on="account_id",
        how="left",
    )

    # --- District features (via account -> district) ---
    district_feats = account[["account_id", "district_id"]].merge(
        district[["district_id", "A11", "A13", "A10", "A4", "A14", "A16"]].rename(
            columns={
                "A11": "district_avg_salary",
                "A13": "district_unemployment_96",
                "A10": "district_urban_ratio",
                "A4": "district_inhabitants",
                "A14": "district_entrepreneurs",
                "A16": "district_crimes_96",
            }
        ),
        on="district_id",
        how="left",
    )
    result = result.merge(
        district_feats.drop(columns=["district_id"]),
        on="account_id",
        how="left",
    )

    # --- Loan features (temporal constraint: loan.date <= rollup_end) ---
    loan_with_end = loan.merge(
        study_table[["account_id", "rollup_end"]], on="account_id", how="inner"
    )
    loan_valid = loan_with_end[
        loan_with_end["date"] <= loan_with_end["rollup_end"]
    ].copy()

    loan_feats = (
        loan_valid.groupby("account_id")
        .agg(
            loan_amount=("amount", "sum"),
            loan_monthly_payment=("payments", "sum"),
        )
        .reset_index()
    )
    loan_feats["has_loan"] = 1

    result = result.merge(loan_feats, on="account_id", how="left")
    result["has_loan"] = result["has_loan"].fillna(0).astype(int)
    result["loan_amount"] = result["loan_amount"].fillna(0)
    result["loan_monthly_payment"] = result["loan_monthly_payment"].fillna(0)

    # --- Order features ---
    order_per_account = (
        order.groupby("account_id")
        .agg(
            order_count=("order_id", "count"),
            order_total_amount=("amount", "sum"),
        )
        .reset_index()
    )

    # Category flags from k_symbol
    order_flags = (
        order.groupby("account_id")["k_symbol"]
        .apply(
            lambda x: pd.Series(
                {
                    "has_insurance_order": int("insurance_payment" in x.values),
                    "has_household_order": int("household_payment" in x.values),
                    "has_loan_order": int("loan_payment" in x.values),
                }
            )
        )
        .unstack(fill_value=0)
        .reset_index()
    )

    order_feats = order_per_account.merge(order_flags, on="account_id", how="left")
    result = result.merge(order_feats, on="account_id", how="left")
    result["order_count"] = result["order_count"].fillna(0).astype(int)
    result["order_total_amount"] = result["order_total_amount"].fillna(0)
    for col in ["has_insurance_order", "has_household_order", "has_loan_order"]:
        result[col] = result[col].fillna(0).astype(int)

    # --- Account features ---
    # Frequency one-hot
    account_freq = account[["account_id", "frequency"]].copy()
    freq_dummies = pd.get_dummies(account_freq["frequency"], prefix="freq").astype(int)
    # Standardize column names
    freq_col_map = {
        "freq_monthly": "freq_monthly",
        "freq_weekly": "freq_weekly",
        "freq_after transaction": "freq_after_transaction",
    }
    freq_dummies = freq_dummies.rename(columns=freq_col_map)
    account_freq = pd.concat([account_freq[["account_id"]], freq_dummies], axis=1)
    result = result.merge(account_freq, on="account_id", how="left")

    # months_to_event from study_table
    result = result.merge(
        study_table[["account_id", "months_to_event"]], on="account_id", how="left"
    )

    return result


# ── Master Builder ───────────────────────────────────────────────────────────


def build_feature_matrix(
    study_table: pd.DataFrame,
    trans: pd.DataFrame,
    client: pd.DataFrame,
    disp: pd.DataFrame,
    account: pd.DataFrame,
    district: pd.DataFrame,
    loan: pd.DataFrame,
    order: pd.DataFrame,
    include_tsfresh: bool = True,
    fc_parameters: dict | None = None,
) -> pd.DataFrame:
    """
    Build the complete feature matrix (one row per account).

    Steps:
      1. Filter transactions to rollup windows
      2. Compute tsfresh features (optional)
      3. Compute baseline features (5 required)
      4. Compute static features (district, loan, order, account)
      5. Left-join all onto study_table[account_id, target, split]
      6. Fill NaN for counts/flags with 0
      7. Validate: one row per account_id

    Returns
    -------
    DataFrame with account_id, target, split, and all feature columns.
    """
    # 1. Filter transactions
    print("Filtering transactions to rollup windows...")
    trans_rollup = filter_transactions_to_rollup(trans, study_table)
    print(f"  {len(trans)} -> {len(trans_rollup)} transactions in rollup windows")

    # 2. tsfresh features
    if include_tsfresh:
        print("Computing tsfresh features (this may take a few minutes)...")
        tsfresh_feats = compute_tsfresh_features(
            trans_rollup, study_table, fc_parameters
        )
        print(f"  {tsfresh_feats.shape[1]} tsfresh features extracted")
    else:
        tsfresh_feats = None

    # 3. Baseline features
    print("Computing baseline features...")
    baseline_feats = compute_baseline_features(
        trans_rollup, study_table, client, disp, account, district
    )

    # 4. Static features
    print("Computing static features...")
    static_feats = compute_static_features(
        study_table, client, disp, account, district, loan, order
    )

    # 5. Assemble
    print("Assembling feature matrix...")
    result = study_table[["account_id", "target", "split"]].copy()

    # Merge baseline (balance_last, turnover come from here)
    baseline_cols = [c for c in baseline_feats.columns if c != "account_id"]
    result = result.merge(baseline_feats, on="account_id", how="left")

    # Merge static (avoid duplicate columns already from baseline)
    static_only_cols = [
        c for c in static_feats.columns if c not in result.columns or c == "account_id"
    ]
    result = result.merge(static_feats[static_only_cols], on="account_id", how="left")

    # Merge tsfresh
    if tsfresh_feats is not None:
        result = result.merge(
            tsfresh_feats, left_on="account_id", right_index=True, how="left"
        )

    # 6. Fill NaN for counts/flags
    flag_cols = [
        "has_loan",
        "has_insurance_order",
        "has_household_order",
        "has_loan_order",
        "order_count",
    ]
    for col in flag_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)

    amount_cols = ["order_total_amount", "loan_amount", "loan_monthly_payment"]
    for col in amount_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    # 7. Validate
    assert result["account_id"].is_unique, "Duplicate account_ids in feature matrix!"
    assert result["target"].notna().all(), "NaN in target column!"
    assert result["split"].notna().all(), "NaN in split column!"

    n_features = len(
        [c for c in result.columns if c not in ("account_id", "target", "split")]
    )
    print(f"Feature matrix: {result.shape[0]} accounts x {n_features} features")

    return result


# ── Feature Overview (Deliverable) ──────────────────────────────────────────


def get_feature_overview_table(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Generate an overview table of all predictive features.

    Returns DataFrame with: feature_name, category, dtype, non_null, mean, std, min, max.
    """
    excluded = {"account_id", "target", "split"}
    feat_cols = [c for c in feature_matrix.columns if c not in excluded]

    rows = []
    for col in feat_cols:
        series = feature_matrix[col]

        # Determine category
        if col.startswith(TSFRESH_FEATURE_PREFIX):
            category = "tsfresh (auto)"
        elif col in BASELINE_FEATURE_COLS:
            category = "baseline"
        elif col.startswith("district_"):
            category = "district"
        elif col.startswith("has_loan") or col.startswith("loan_"):
            category = "loan"
        elif col.startswith("order_") or col.startswith("has_") and "order" in col:
            category = "order"
        elif col.startswith("freq_"):
            category = "account"
        elif col == "months_to_event":
            category = "account"
        else:
            category = "other"

        row = {
            "feature_name": col,
            "category": category,
            "dtype": str(series.dtype),
            "non_null": series.notna().sum(),
            "pct_null": round(series.isna().mean() * 100, 1),
        }

        if pd.api.types.is_numeric_dtype(series):
            row.update(
                {
                    "mean": round(series.mean(), 2),
                    "std": round(series.std(), 2),
                    "min": round(series.min(), 2),
                    "max": round(series.max(), 2),
                }
            )

        rows.append(row)

    overview = pd.DataFrame(rows)
    return overview
