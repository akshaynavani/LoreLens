import math

from lorelens.evals import metrics

RETRIEVED = [
    "https://docs.example.com/nimbus/v2/api/rest-endpoints.md",
    "nimbus/v2/guides/authentication.md",
    None,
]


def test_hit_rate_and_mrr():
    expected = ["nimbus/v2/guides/authentication.md"]
    assert metrics.hit_rate_at_k(RETRIEVED, expected, k=1) == 0.0
    assert metrics.hit_rate_at_k(RETRIEVED, expected, k=2) == 1.0
    assert metrics.reciprocal_rank(RETRIEVED, expected) == 0.5


def test_url_suffix_matching():
    assert metrics.reciprocal_rank(RETRIEVED, ["nimbus/v2/api/rest-endpoints.md"]) == 1.0


def test_recall_and_citation_precision():
    expected = ["nimbus/v2/guides/authentication.md", "nimbus/v1/guides/authentication.md"]
    assert metrics.recall(RETRIEVED, expected) == 0.5
    assert metrics.citation_precision(["nimbus/v2/guides/authentication.md", "x.md"], expected) == 0.5


def test_no_expected_sources_is_nan_and_ignored_in_mean():
    assert math.isnan(metrics.hit_rate_at_k(RETRIEVED, [], k=3))
    assert metrics.mean([1.0, math.nan, 0.0]) == 0.5


def test_percentile_interpolates():
    assert metrics.percentile([10, 20, 30, 40], 50) == 25
    assert metrics.percentile([5], 95) == 5
