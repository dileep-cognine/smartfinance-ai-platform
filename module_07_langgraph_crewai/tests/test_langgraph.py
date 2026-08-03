from src.langgraph.graph_builder import review_route


def test_approved_review_finalizes() -> None:
    assert review_route({"review_comments": "APPROVED", "revision_count": 1}) == "finalize"


def test_rejected_review_revises() -> None:
    assert review_route({"review_comments": "Add citations", "revision_count": 1}) == "revise"


def test_third_rejection_escalates() -> None:
    assert (
        review_route({"review_comments": "Still incomplete", "revision_count": 3})
        == "human_review"
    )
