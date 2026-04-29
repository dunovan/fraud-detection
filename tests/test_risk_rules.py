from risk_rules import label_risk, score_transaction


def _base_tx(**overrides):
    tx = {
        "device_risk_score": 10,
        "is_international": 0,
        "amount_usd": 50,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


# --- label_risk thresholds ---

def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


# --- individual signal tests ---

def test_large_amount_adds_risk():
    assert score_transaction(_base_tx(amount_usd=1200)) >= 25


def test_moderate_amount_adds_risk():
    assert score_transaction(_base_tx(amount_usd=600)) >= 10


def test_high_device_risk_adds_score():
    assert score_transaction(_base_tx(device_risk_score=75)) >= 25


def test_moderate_device_risk_adds_score():
    assert score_transaction(_base_tx(device_risk_score=50)) >= 10


def test_international_adds_score():
    assert score_transaction(_base_tx(is_international=1)) >= 15


def test_high_velocity_adds_score():
    assert score_transaction(_base_tx(velocity_24h=7)) >= 20


def test_moderate_velocity_adds_score():
    assert score_transaction(_base_tx(velocity_24h=4)) >= 5


def test_login_failures_high_adds_score():
    assert score_transaction(_base_tx(failed_logins_24h=6)) >= 20


def test_login_failures_moderate_adds_score():
    assert score_transaction(_base_tx(failed_logins_24h=3)) >= 10


def test_one_prior_chargeback_adds_score():
    assert score_transaction(_base_tx(prior_chargebacks=1)) >= 10


def test_multiple_prior_chargebacks_add_more_score():
    one = score_transaction(_base_tx(prior_chargebacks=1))
    two = score_transaction(_base_tx(prior_chargebacks=2))
    assert two > one
    assert two >= 20


# --- end-to-end label tests ---

def test_clean_transaction_is_low_risk():
    # Low amount, domestic, low velocity, trusted device, no history
    assert label_risk(score_transaction(_base_tx())) == "low"


def test_high_risk_transaction_is_labeled_high():
    # Matches profile of confirmed chargebacks in the dataset
    tx = _base_tx(
        device_risk_score=81,
        is_international=1,
        amount_usd=1250,
        velocity_24h=6,
        failed_logins_24h=5,
    )
    assert label_risk(score_transaction(tx)) == "high"


def test_score_clamped_at_100():
    # All signals firing should not exceed 100
    tx = _base_tx(
        device_risk_score=90,
        is_international=1,
        amount_usd=2000,
        velocity_24h=10,
        failed_logins_24h=8,
        prior_chargebacks=3,
    )
    assert score_transaction(tx) == 100
