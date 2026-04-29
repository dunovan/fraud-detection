from pathlib import Path

import pandas as pd
import pytest

from analyze_fraud import score_transactions, summarize_results


# --- helpers ---

def _txn(transaction_id, account_id, **overrides):
    row = {
        "transaction_id": transaction_id,
        "account_id": account_id,
        "amount_usd": 50.0,
        "device_risk_score": 10,
        "is_international": 0,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
    }
    row.update(overrides)
    return row


def _acct(account_id, prior_chargebacks=0):
    return {"account_id": account_id, "prior_chargebacks": prior_chargebacks}


def _scored_df(rows):
    """Minimal pre-scored DataFrame for summarize_results tests.

    Each row is (transaction_id, risk_label, amount_usd).
    """
    return pd.DataFrame(
        [{"transaction_id": r[0], "risk_label": r[1], "amount_usd": r[2]} for r in rows]
    )


def _chargebacks_df(*ids):
    return pd.DataFrame({"transaction_id": list(ids)})


# --- score_transactions ---

def test_score_transactions_adds_risk_score_column():
    txns = pd.DataFrame([_txn(1, 101)])
    accts = pd.DataFrame([_acct(101)])
    result = score_transactions(txns, accts)
    assert "risk_score" in result.columns


def test_score_transactions_adds_risk_label_column():
    txns = pd.DataFrame([_txn(1, 101)])
    accts = pd.DataFrame([_acct(101)])
    result = score_transactions(txns, accts)
    assert "risk_label" in result.columns


def test_score_transactions_low_risk_profile():
    txns = pd.DataFrame([_txn(1, 101)])
    accts = pd.DataFrame([_acct(101)])
    result = score_transactions(txns, accts)
    assert result["risk_label"].iloc[0] == "low"


def test_score_transactions_high_risk_profile():
    txns = pd.DataFrame([_txn(
        1, 101,
        device_risk_score=81, is_international=1,
        amount_usd=1250, velocity_24h=6, failed_logins_24h=5,
    )])
    accts = pd.DataFrame([_acct(101)])
    result = score_transactions(txns, accts)
    assert result["risk_label"].iloc[0] == "high"


def test_score_transactions_uses_account_prior_chargebacks():
    txns = pd.DataFrame([_txn(1, 101), _txn(2, 102)])
    accts = pd.DataFrame([_acct(101, prior_chargebacks=0), _acct(102, prior_chargebacks=2)])
    result = score_transactions(txns, accts)
    score_clean = result.loc[result["transaction_id"] == 1, "risk_score"].iloc[0]
    score_risky = result.loc[result["transaction_id"] == 2, "risk_score"].iloc[0]
    assert score_risky > score_clean


def test_score_transactions_preserves_row_count():
    txns = pd.DataFrame([_txn(i, 101) for i in range(5)])
    accts = pd.DataFrame([_acct(101)])
    result = score_transactions(txns, accts)
    assert len(result) == 5


def test_score_transactions_risk_score_in_valid_range():
    txns = pd.DataFrame([_txn(1, 101)])
    accts = pd.DataFrame([_acct(101)])
    result = score_transactions(txns, accts)
    assert 0 <= result["risk_score"].iloc[0] <= 100


# --- summarize_results ---

def test_summarize_results_has_required_columns():
    scored = _scored_df([(1, "low", 50.0), (2, "high", 500.0)])
    summary = summarize_results(scored, _chargebacks_df(2))
    for col in ["risk_label", "transactions", "total_amount_usd", "avg_amount_usd",
                "chargebacks", "chargeback_rate"]:
        assert col in summary.columns


def test_summarize_results_transaction_counts():
    scored = _scored_df([(1, "low", 50.0), (2, "low", 30.0), (3, "high", 500.0)])
    summary = summarize_results(scored, _chargebacks_df())
    low = summary.loc[summary["risk_label"] == "low", "transactions"].iloc[0]
    high = summary.loc[summary["risk_label"] == "high", "transactions"].iloc[0]
    assert low == 2
    assert high == 1


def test_summarize_results_total_amount():
    scored = _scored_df([(1, "low", 100.0), (2, "low", 200.0)])
    summary = summarize_results(scored, _chargebacks_df())
    total = summary.loc[summary["risk_label"] == "low", "total_amount_usd"].iloc[0]
    assert total == 300.0


def test_summarize_results_chargeback_rate_full_capture():
    scored = _scored_df([(1, "high", 500.0), (2, "high", 600.0)])
    summary = summarize_results(scored, _chargebacks_df(1, 2))
    rate = summary.loc[summary["risk_label"] == "high", "chargeback_rate"].iloc[0]
    assert rate == 1.0


def test_summarize_results_chargeback_rate_zero_for_clean_bucket():
    scored = _scored_df([(1, "low", 50.0), (2, "high", 500.0)])
    summary = summarize_results(scored, _chargebacks_df(2))
    rate = summary.loc[summary["risk_label"] == "low", "chargeback_rate"].iloc[0]
    assert rate == 0.0


def test_summarize_results_partial_chargeback_rate():
    scored = _scored_df([(1, "high", 500.0), (2, "high", 600.0), (3, "high", 700.0)])
    summary = summarize_results(scored, _chargebacks_df(1))
    rate = summary.loc[summary["risk_label"] == "high", "chargeback_rate"].iloc[0]
    assert abs(rate - 1 / 3) < 1e-9


# --- integration tests against the real dataset ---

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def pipeline_output():
    accounts = pd.read_csv(DATA_DIR / "accounts.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    chargebacks = pd.read_csv(DATA_DIR / "chargebacks.csv")
    scored = score_transactions(transactions, accounts)
    summary = summarize_results(scored, chargebacks)
    return scored, chargebacks, summary


def test_no_confirmed_fraud_labeled_low_risk(pipeline_output):
    scored, chargebacks, _ = pipeline_output
    fraud_ids = set(chargebacks["transaction_id"])
    low_fraud = scored[(scored["transaction_id"].isin(fraud_ids)) & (scored["risk_label"] == "low")]
    assert len(low_fraud) == 0, (
        f"Confirmed fraud transactions scored low risk: {low_fraud['transaction_id'].tolist()}"
    )


def test_high_bucket_chargeback_rate_exceeds_low_bucket(pipeline_output):
    _, _, summary = pipeline_output
    high_rate = summary.loc[summary["risk_label"] == "high", "chargeback_rate"].iloc[0]
    low_rate = summary.loc[summary["risk_label"] == "low", "chargeback_rate"].iloc[0]
    assert high_rate > low_rate


def test_low_bucket_contains_zero_confirmed_fraud(pipeline_output):
    _, _, summary = pipeline_output
    low_row = summary[summary["risk_label"] == "low"]
    assert not low_row.empty
    assert low_row["chargebacks"].iloc[0] == 0


def test_all_transactions_scored(pipeline_output):
    scored, _, _ = pipeline_output
    accounts = pd.read_csv(DATA_DIR / "accounts.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    assert len(scored) == len(transactions)
    assert scored["risk_score"].notna().all()
    assert scored["risk_label"].notna().all()


def test_risk_labels_are_valid_values(pipeline_output):
    scored, _, _ = pipeline_output
    assert set(scored["risk_label"]).issubset({"low", "medium", "high"})
