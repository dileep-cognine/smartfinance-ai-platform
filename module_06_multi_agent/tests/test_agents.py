from fastapi.testclient import TestClient

from src.communication.messaging import AgentMessage, ResearchState
from src.main import app


def test_message_schema() -> None:
    message = AgentMessage(
        run_id="run",
        sender="data",
        recipient="orchestrator",
        status="success",
    )
    assert message.payload == {}


def test_research_state_normalizes_defaults() -> None:
    state = ResearchState(run_id="run", ticker="MSFT")
    assert state.status == "running"
    assert state.final_report is None


def test_health_endpoint() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200


def test_invalid_ticker_is_rejected_without_network_call() -> None:
    response = TestClient(app).post("/research/start", json={"ticker": "$$$"})
    assert response.status_code == 422
