from common.utils.validation_utils import validate_ticker


def test_validate_ticker_normalizes() -> None:
    assert validate_ticker(" msft ") == "MSFT"
