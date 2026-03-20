"""
Feature engineering for the credit card affinity model.

Provides:
  - Transaction rollup filtering
  - tsfresh-based automatic feature extraction
  - Behavioral features from transaction type, operation, k_symbol
  - Window comparison features (temporal sub-windows within rollup)
  - tsfresh deduplication by correlation
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

BEHAVIORAL_FEATURE_COLS = [
    # Group A — Volume & Activity
    "n_transactions",
    "n_active_days",
    "avg_amount",
    "std_amount",
    "median_amount",
    "balance_range",
    # Group B — Transaction Type Ratios
    "pct_type_credit",
    "pct_type_withdrawal",
    "credit_withdrawal_ratio",
    "avg_credit_amount",
    # Group C — Operation Patterns
    "pct_op_collection_from_another_bank",
    "pct_op_credit_in_cash",
    "pct_op_withdrawal_in_cash",
    "pct_op_remittance_to_another_bank",
    "pct_op_credit_card_withdrawal",
    # Group D — K_symbol Purpose
    "pct_ksym_old_age_pension",
    "pct_ksym_insurance_payment",
    "pct_ksym_household_payment",
    "pct_ksym_loan_payment",
    "n_distinct_ksymbols",
    # Group E — Temporal Patterns
    "avg_days_between_tx",
    "std_days_between_tx",
    "pct_tx_first_week",
    "pct_tx_last_week",
    "n_distinct_banks",
    "has_external_bank_tx",
    # Group F — Derived Ratios
    "turnover_per_transaction",
    "balance_to_turnover_ratio",
    "net_flow",
    "net_flow_ratio",
]

WINDOW_FEATURE_COLS = [
    "balance_mean_q1",
    "balance_mean_q4",
    "balance_trend_q1_q4",
    "balance_trend_q1_q4_pct",
    "turnover_h1",
    "turnover_h2",
    "turnover_trend_h1_h2",
    "turnover_trend_h1_h2_pct",
    "n_tx_h1",
    "n_tx_h2",
    "tx_count_trend_h1_h2",
    "monthly_balance_slope",
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


# ── Behavioral Features (30) ────────────────────────────────────────────────


def compute_behavioral_features(
    trans_rollup: pd.DataFrame,
    study_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute 30 behavioral features from transaction details (type, operation,
    k_symbol, bank, temporal patterns, derived ratios).

    Parameters
    ----------
    trans_rollup : Rollup-filtered transactions with columns:
        account_id, date, amount, balance, type, operation, k_symbol, bank
    study_table : Study table with account_id

    Returns
    -------
    DataFrame with account_id + 30 behavioral feature columns.
    """
    result = study_table[["account_id"]].copy()
    tr = trans_rollup.copy()

    # --- Group A: Volume & Activity ---
    grp = tr.groupby("account_id")
    vol = pd.DataFrame(index=grp.groups.keys())
    vol.index.name = "account_id"
    vol["n_transactions"] = grp.size()
    vol["n_active_days"] = grp["date"].nunique()
    vol["avg_amount"] = grp["amount"].apply(lambda x: x.abs().mean())
    vol["std_amount"] = grp["amount"].apply(lambda x: x.abs().std())
    vol["median_amount"] = grp["amount"].apply(lambda x: x.abs().median())
    vol["balance_range"] = grp["balance"].max() - grp["balance"].min()
    vol = vol.reset_index()

    result = result.merge(vol, on="account_id", how="left")

    # --- Group B: Transaction Type Ratios ---
    type_counts = tr.groupby("account_id")["type"].value_counts(normalize=True).unstack(fill_value=0)
    for col_name, type_val in [("pct_type_credit", "credit"), ("pct_type_withdrawal", "withdrawal")]:
        if type_val in type_counts.columns:
            type_counts = type_counts.rename(columns={type_val: col_name})
        else:
            type_counts[col_name] = 0.0

    # credit_withdrawal_ratio
    raw_type_counts = tr.groupby("account_id")["type"].value_counts().unstack(fill_value=0)
    n_credit = raw_type_counts.get("credit", pd.Series(0, index=raw_type_counts.index))
    n_withdrawal = raw_type_counts.get("withdrawal", pd.Series(0, index=raw_type_counts.index))
    type_ratios = pd.DataFrame({
        "pct_type_credit": type_counts.get("pct_type_credit", 0),
        "pct_type_withdrawal": type_counts.get("pct_type_withdrawal", 0),
        "credit_withdrawal_ratio": n_credit / (n_withdrawal + 1),
    })
    type_ratios.index.name = "account_id"

    # avg_credit_amount
    credit_tx = tr[tr["type"] == "credit"]
    avg_credit = credit_tx.groupby("account_id")["amount"].apply(lambda x: x.abs().mean()).rename("avg_credit_amount")
    type_ratios = type_ratios.merge(avg_credit, left_index=True, right_index=True, how="left")
    type_ratios["avg_credit_amount"] = type_ratios["avg_credit_amount"].fillna(0)

    type_ratios = type_ratios.reset_index()
    result = result.merge(type_ratios, on="account_id", how="left")

    # --- Group C: Operation Patterns ---
    op_pcts = tr.groupby("account_id")["operation"].value_counts(normalize=True).unstack(fill_value=0)
    op_mapping = {
        "collection_from_another_bank": "pct_op_collection_from_another_bank",
        "credit_in_cash": "pct_op_credit_in_cash",
        "withdrawal_in_cash": "pct_op_withdrawal_in_cash",
        "remittance_to_another_bank": "pct_op_remittance_to_another_bank",
        "credit_card_withdrawal": "pct_op_credit_card_withdrawal",
    }
    op_feats = pd.DataFrame(index=op_pcts.index)
    op_feats.index.name = "account_id"
    for raw_name, feat_name in op_mapping.items():
        op_feats[feat_name] = op_pcts.get(raw_name, 0)
    op_feats = op_feats.reset_index()
    result = result.merge(op_feats, on="account_id", how="left")

    # --- Group D: K_symbol Purpose ---
    # Filter to non-empty k_symbol values
    tr_ksym = tr[tr["k_symbol"].notna() & (tr["k_symbol"] != "")].copy()

    ksym_pcts = tr_ksym.groupby("account_id")["k_symbol"].value_counts(normalize=True).unstack(fill_value=0)
    ksym_mapping = {
        "old_age_pension": "pct_ksym_old_age_pension",
        "insurance_payment": "pct_ksym_insurance_payment",
        "household_payment": "pct_ksym_household_payment",
        "loan_payment": "pct_ksym_loan_payment",
    }
    ksym_feats = pd.DataFrame(index=study_table["account_id"])
    ksym_feats.index.name = "account_id"
    for raw_name, feat_name in ksym_mapping.items():
        if raw_name in ksym_pcts.columns:
            ksym_feats[feat_name] = ksym_pcts[raw_name]
        else:
            ksym_feats[feat_name] = 0.0
    ksym_feats = ksym_feats.fillna(0)

    # n_distinct_ksymbols (count of distinct non-empty k_symbol per account)
    n_ksym = tr_ksym.groupby("account_id")["k_symbol"].nunique().rename("n_distinct_ksymbols")
    ksym_feats = ksym_feats.merge(n_ksym, left_index=True, right_index=True, how="left")
    ksym_feats["n_distinct_ksymbols"] = ksym_feats["n_distinct_ksymbols"].fillna(0).astype(int)
    ksym_feats = ksym_feats.reset_index()
    result = result.merge(ksym_feats, on="account_id", how="left")

    # --- Group E: Temporal Patterns ---
    sorted_tr = tr.sort_values(["account_id", "date"])

    # Days between consecutive transactions
    sorted_tr["prev_date"] = sorted_tr.groupby("account_id")["date"].shift(1)
    sorted_tr["days_between"] = (sorted_tr["date"] - sorted_tr["prev_date"]).dt.days

    days_stats = sorted_tr.groupby("account_id")["days_between"].agg(
        avg_days_between_tx="mean",
        std_days_between_tx="std",
    ).reset_index()
    days_stats["std_days_between_tx"] = days_stats["std_days_between_tx"].fillna(0)

    # % transactions in first/last week of month
    sorted_tr["day_of_month"] = sorted_tr["date"].dt.day
    week_feats = sorted_tr.groupby("account_id").apply(
        lambda g: pd.Series({
            "pct_tx_first_week": (g["day_of_month"] <= 7).mean(),
            "pct_tx_last_week": (g["day_of_month"] >= 24).mean(),
        }),
        include_groups=False,
    ).reset_index()

    # Bank diversity
    bank_feats = tr.groupby("account_id").apply(
        lambda g: pd.Series({
            "n_distinct_banks": g["bank"].dropna().replace("", np.nan).dropna().nunique(),
            "has_external_bank_tx": int(g["bank"].dropna().replace("", np.nan).dropna().shape[0] > 0),
        }),
        include_groups=False,
    ).reset_index()

    result = result.merge(days_stats, on="account_id", how="left")
    result = result.merge(week_feats, on="account_id", how="left")
    result = result.merge(bank_feats, on="account_id", how="left")

    # --- Group F: Derived Ratios ---
    # Need turnover and balance_last per account
    acct_agg = tr.groupby("account_id").apply(
        lambda g: pd.Series({
            "turnover_per_transaction": g["amount"].abs().sum() / len(g) if len(g) > 0 else 0,
            "balance_to_turnover_ratio": (
                g.sort_values("date")["balance"].iloc[-1] / (g["amount"].abs().sum() + 1)
            ),
            "net_flow": (
                g.loc[g["type"] == "credit", "amount"].abs().sum()
                - g.loc[g["type"] == "withdrawal", "amount"].abs().sum()
            ),
        }),
        include_groups=False,
    ).reset_index()
    acct_agg["net_flow_ratio"] = acct_agg["net_flow"] / (
        acct_agg.get("turnover_per_transaction", 1) * result.set_index("account_id").reindex(acct_agg["account_id"])["n_transactions"].values + 1
    )

    # Recalculate net_flow_ratio properly: net_flow / (turnover + 1)
    turnover_series = tr.groupby("account_id")["amount"].apply(lambda x: x.abs().sum())
    acct_agg = acct_agg.set_index("account_id")
    acct_agg["net_flow_ratio"] = acct_agg["net_flow"] / (turnover_series + 1)
    acct_agg = acct_agg.reset_index()

    result = result.merge(
        acct_agg[["account_id", "turnover_per_transaction", "balance_to_turnover_ratio", "net_flow", "net_flow_ratio"]],
        on="account_id",
        how="left",
    )

    # Fill NaN for accounts with no transactions
    for col in BEHAVIORAL_FEATURE_COLS:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    return result


# ── Window Comparison Features (12) ─────────────────────────────────────────


def compute_window_comparison_features(
    trans_rollup: pd.DataFrame,
    study_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute 12 temporal sub-window features comparing different periods
    within each account's 12-month rollup window.

    Sub-windows:
      - Q1-Q4: each 3 months (Q1 = oldest, Q4 = most recent)
      - H1/H2: each 6 months

    Parameters
    ----------
    trans_rollup : Rollup-filtered transactions
    study_table : Study table with account_id, rollup_start

    Returns
    -------
    DataFrame with account_id + 12 window comparison features.
    """
    result = study_table[["account_id"]].copy()

    # Merge rollup_start to assign sub-windows
    tr = trans_rollup.merge(
        study_table[["account_id", "rollup_start"]], on="account_id", how="left"
    )
    tr["months_elapsed"] = (tr["date"] - tr["rollup_start"]).dt.days / 30.44

    # Assign quarters (Q1=0-3, Q2=3-6, Q3=6-9, Q4=9-12)
    tr["quarter"] = pd.cut(
        tr["months_elapsed"],
        bins=[0, 3, 6, 9, 12.1],
        labels=["Q1", "Q2", "Q3", "Q4"],
        include_lowest=True,
    )
    # Assign halves (H1=0-6, H2=6-12)
    tr["half"] = pd.cut(
        tr["months_elapsed"],
        bins=[0, 6, 12.1],
        labels=["H1", "H2"],
        include_lowest=True,
    )

    # --- Balance by quarter ---
    balance_by_q = tr.groupby(["account_id", "quarter"], observed=True)["balance"].mean().unstack(fill_value=np.nan)
    q_feats = pd.DataFrame(index=study_table["account_id"])
    q_feats.index.name = "account_id"
    q_feats["balance_mean_q1"] = balance_by_q.get("Q1", np.nan)
    q_feats["balance_mean_q4"] = balance_by_q.get("Q4", np.nan)
    q_feats["balance_trend_q1_q4"] = q_feats["balance_mean_q4"] - q_feats["balance_mean_q1"]
    q_feats["balance_trend_q1_q4_pct"] = q_feats["balance_trend_q1_q4"] / (q_feats["balance_mean_q1"].abs() + 1)

    # --- Turnover and tx count by half ---
    turnover_by_h = tr.groupby(["account_id", "half"], observed=True)["amount"].apply(
        lambda x: x.abs().sum()
    ).unstack(fill_value=0)
    tx_count_by_h = tr.groupby(["account_id", "half"], observed=True).size().unstack(fill_value=0)

    h_feats = pd.DataFrame(index=study_table["account_id"])
    h_feats.index.name = "account_id"
    h_feats["turnover_h1"] = turnover_by_h.get("H1", 0)
    h_feats["turnover_h2"] = turnover_by_h.get("H2", 0)
    h_feats["turnover_trend_h1_h2"] = h_feats["turnover_h2"] - h_feats["turnover_h1"]
    h_feats["turnover_trend_h1_h2_pct"] = h_feats["turnover_trend_h1_h2"] / (h_feats["turnover_h1"] + 1)
    h_feats["n_tx_h1"] = tx_count_by_h.get("H1", 0)
    h_feats["n_tx_h2"] = tx_count_by_h.get("H2", 0)
    h_feats["tx_count_trend_h1_h2"] = h_feats["n_tx_h2"] - h_feats["n_tx_h1"]

    # --- Monthly balance slope (OLS) ---
    tr["month_idx"] = (tr["months_elapsed"]).clip(lower=0).astype(int).clip(upper=11)
    monthly_balance = tr.groupby(["account_id", "month_idx"])["balance"].mean().reset_index()

    def _compute_slope(group):
        if len(group) < 2:
            return 0.0
        coeffs = np.polyfit(group["month_idx"].values, group["balance"].values, deg=1)
        return coeffs[0]

    slopes = monthly_balance.groupby("account_id").apply(_compute_slope, include_groups=False).rename("monthly_balance_slope")

    # --- Combine ---
    q_feats = q_feats.reset_index()
    h_feats = h_feats.reset_index()
    slopes = slopes.reset_index()

    result = result.merge(q_feats, on="account_id", how="left")
    result = result.merge(h_feats, on="account_id", how="left")
    result = result.merge(slopes, on="account_id", how="left")

    # Fill NaN: 0 for counts/turnover, NaN→0 for trends and slope
    for col in WINDOW_FEATURE_COLS:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    return result


# ── tsfresh Deduplication by Correlation ─────────────────────────────────────


def select_tsfresh_by_correlation(
    tsfresh_df: pd.DataFrame,
    y: pd.Series,
    corr_threshold: float = 0.95,
) -> list[str]:
    """
    Select tsfresh features by removing highly correlated redundant features.

    Algorithm:
      1. Compute abs(correlation) of each tsfresh feature with target
      2. Sort descending by correlation
      3. Greedily keep a feature only if its Pearson correlation with ALL
         already-kept features is below corr_threshold
      4. Return list of deduplicated column names

    Parameters
    ----------
    tsfresh_df : DataFrame of tsfresh features (columns to select from)
    y : Target series aligned with tsfresh_df index
    corr_threshold : Maximum allowed pairwise correlation (default 0.95)

    Returns
    -------
    List of selected column names (~100-200 features).
    """
    # Align indices
    common_idx = tsfresh_df.index.intersection(y.index)
    X = tsfresh_df.loc[common_idx]
    y_aligned = y.loc[common_idx]

    # Correlation with target
    target_corr = X.corrwith(y_aligned).abs().sort_values(ascending=False)
    target_corr = target_corr.dropna()

    selected = []
    selected_data = pd.DataFrame(index=X.index)

    for col in target_corr.index:
        if len(selected) == 0:
            selected.append(col)
            selected_data[col] = X[col]
            continue

        # Check correlation with all already-selected features
        corrs = selected_data.corrwith(X[col]).abs()
        if (corrs < corr_threshold).all():
            selected.append(col)
            selected_data[col] = X[col]

    return selected


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
    tsfresh_corr_threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Build the complete feature matrix (one row per account).

    Steps:
      1. Filter transactions to rollup windows
      2. Compute tsfresh features (optional)
      3. Compute baseline features (5 required)
      4. Compute static features (district, loan, order, account)
      4a. Compute behavioral features (30 transaction-detail features)
      4b. Compute window comparison features (12 temporal sub-window features)
      4c. Deduplicate tsfresh features by correlation (if include_tsfresh)
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

    # 4a. Behavioral features
    print("Computing behavioral features (30 features)...")
    behavioral_feats = compute_behavioral_features(trans_rollup, study_table)
    print(f"  {len([c for c in behavioral_feats.columns if c != 'account_id'])} behavioral features computed")

    # 4b. Window comparison features
    print("Computing window comparison features (12 features)...")
    window_feats = compute_window_comparison_features(trans_rollup, study_table)
    print(f"  {len([c for c in window_feats.columns if c != 'account_id'])} window features computed")

    # 4c. Deduplicate tsfresh features by correlation
    if tsfresh_feats is not None:
        print(f"Deduplicating tsfresh features (corr_threshold={tsfresh_corr_threshold})...")
        y_for_corr = study_table.set_index("account_id")["target"]
        selected_tsf_cols = select_tsfresh_by_correlation(
            tsfresh_feats, y_for_corr, corr_threshold=tsfresh_corr_threshold
        )
        print(f"  {tsfresh_feats.shape[1]} -> {len(selected_tsf_cols)} tsfresh features after deduplication")
        tsfresh_feats = tsfresh_feats[selected_tsf_cols]

    # 5. Assemble
    print("Assembling feature matrix...")
    result = study_table[["account_id", "target", "split"]].copy()

    # Merge baseline (balance_last, turnover come from here)
    result = result.merge(baseline_feats, on="account_id", how="left")

    # Merge static (avoid duplicate columns already from baseline)
    static_only_cols = [
        c for c in static_feats.columns if c not in result.columns or c == "account_id"
    ]
    result = result.merge(static_feats[static_only_cols], on="account_id", how="left")

    # Merge behavioral (avoid duplicate columns)
    behavioral_only_cols = [
        c for c in behavioral_feats.columns if c not in result.columns or c == "account_id"
    ]
    result = result.merge(behavioral_feats[behavioral_only_cols], on="account_id", how="left")

    # Merge window comparison
    window_only_cols = [
        c for c in window_feats.columns if c not in result.columns or c == "account_id"
    ]
    result = result.merge(window_feats[window_only_cols], on="account_id", how="left")

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
        "has_external_bank_tx",
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

    behavioral_set = set(BEHAVIORAL_FEATURE_COLS)
    window_set = set(WINDOW_FEATURE_COLS)

    rows = []
    for col in feat_cols:
        series = feature_matrix[col]

        # Determine category
        if col.startswith(TSFRESH_FEATURE_PREFIX):
            category = "tsfresh (auto)"
        elif col in BASELINE_FEATURE_COLS:
            category = "baseline"
        elif col in behavioral_set:
            category = "behavioral"
        elif col in window_set:
            category = "window"
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
