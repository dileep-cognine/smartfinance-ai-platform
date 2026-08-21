import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.model_registry.registry import promote_if_improved
from src.monitoring import population_stability_index


def test_psi_is_small_for_identical_samples() -> None:
    values = list(range(100))
    assert population_stability_index(values, values) == pytest.approx(0.0)


def test_psi_rejects_small_samples() -> None:
    with pytest.raises(ValueError):
        population_stability_index([1, 2], [1, 2])


def test_drift_endpoint() -> None:
    client = TestClient(app)
    response = client.post(
        "/drift",
        json={
            "reference": {"age": list(range(100))},
            "current": {"age": list(range(100))},
        },
    )
    assert response.status_code == 200
    assert response.json()["drifted_features"] == []


class FakeClient:
    def __init__(self) -> None:
        self.transitions = []
        self.aliases = []

    def transition_model_version_stage(self, name, version, stage, **kwargs):
        self.transitions.append((name, version, stage))

    def set_registered_model_alias(self, name, alias, version):
        self.aliases.append((name, alias, version))


def test_promotion_requires_more_than_two_percent() -> None:
    client = FakeClient()
    assert not promote_if_improved("2", 0.82, 0.80, client=client)
    assert promote_if_improved("2", 0.821, 0.80, client=client)
    assert client.aliases[-1][1] == "champion"
