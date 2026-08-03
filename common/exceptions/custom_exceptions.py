class SmartFinanceError(Exception):
    """Base platform exception."""


class ConfigurationError(SmartFinanceError):
    """Required runtime configuration is invalid."""


class DataValidationError(SmartFinanceError):
    """Input data violates the declared contract."""


class ExternalServiceError(SmartFinanceError):
    """A dependency failed or timed out."""
