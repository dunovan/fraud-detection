import math

import pandas as pd

from features import build_model_frame


def _txns(**overrides):
    row = {"transaction_id": 1, "account_id": 101, "amount_usd": 50.0, "failed_logins_24h": 0}
    row.update(overrides)
    return pd.DataFrame([row])


def _accts(**overrides):
    row = {"account_id": 101, "prior_chargebacks": 0}
    row.update(overrides)
    return pd.DataFrame([row])


# --- merge behaviour ---

def test_account_fields_joined_by_account_id():
    df = build_model_frame(_txns(), _accts(prior_chargebacks=3))
    assert df["prior_chargebacks"].iloc[0] == 3


def test_unknown_account_yields_nan_not_error():
    df = build_model_frame(_txns(account_id=999), _accts(account_id=101))
    assert math.isnan(df["prior_chargebacks"].iloc[0])


def test_output_retains_all_transaction_columns():
    df = build_model_frame(_txns(), _accts())
    assert "transaction_id" in df.columns
    assert "amount_usd" in df.columns


# --- is_large_amount ---

def test_is_large_amount_at_threshold():
    df = build_model_frame(_txns(amount_usd=1000.0), _accts())
    assert df["is_large_amount"].iloc[0] == 1


def test_is_large_amount_just_below_threshold():
    df = build_model_frame(_txns(amount_usd=999.99), _accts())
    assert df["is_large_amount"].iloc[0] == 0


def test_is_large_amount_well_above_threshold():
    df = build_model_frame(_txns(amount_usd=5000.0), _accts())
    assert df["is_large_amount"].iloc[0] == 1


def test_is_large_amount_small_purchase():
    df = build_model_frame(_txns(amount_usd=20.0), _accts())
    assert df["is_large_amount"].iloc[0] == 0


# --- login_pressure ---

def test_login_pressure_none_at_zero_logins():
    df = build_model_frame(_txns(failed_logins_24h=0), _accts())
    assert str(df["login_pressure"].iloc[0]) == "none"


def test_login_pressure_low_at_one_login():
    df = build_model_frame(_txns(failed_logins_24h=1), _accts())
    assert str(df["login_pressure"].iloc[0]) == "low"


def test_login_pressure_low_at_upper_boundary():
    df = build_model_frame(_txns(failed_logins_24h=2), _accts())
    assert str(df["login_pressure"].iloc[0]) == "low"


def test_login_pressure_high_above_boundary():
    df = build_model_frame(_txns(failed_logins_24h=3), _accts())
    assert str(df["login_pressure"].iloc[0]) == "high"


def test_login_pressure_high_at_large_value():
    df = build_model_frame(_txns(failed_logins_24h=10), _accts())
    assert str(df["login_pressure"].iloc[0]) == "high"
