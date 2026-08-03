from src.shap_explanations import natural_language_explanation


def test_natural_language_explanation_names_primary_factors() -> None:
    text = natural_language_explanation(
        {"debt_ratio": 0.7, "age": -0.2, "monthly_income": -0.1},
        {"debt_ratio": 0.42, "age": 35, "monthly_income": 5000},
        denied=True,
    )
    assert "declined" in text
    assert "debt ratio" in text
    assert "review" in text
