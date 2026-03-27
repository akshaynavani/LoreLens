from lorelens.models import SearchFilters


def test_empty_filters_produce_no_pinecone_filter():
    assert SearchFilters().to_pinecone() is None
    assert SearchFilters().is_empty()


def test_single_filter_is_not_wrapped():
    assert SearchFilters(product="nimbus").to_pinecone() == {"product": {"$eq": "nimbus"}}


def test_multiple_filters_use_and():
    f = SearchFilters(product="nimbus", version="v2", tags=["auth"]).to_pinecone()
    assert f == {
        "$and": [
            {"product": {"$eq": "nimbus"}},
            {"version": {"$eq": "v2"}},
            {"tags": {"$in": ["auth"]}},
        ]
    }
