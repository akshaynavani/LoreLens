from lorelens.ingestion.metadata import extract_metadata, normalize_version
from lorelens.models import SourceDocument


def _doc(source: str, text: str = "Body", **meta) -> SourceDocument:
    return SourceDocument(doc_id="d", text=text, source=source, title="T", metadata=meta)


def test_product_version_and_type_from_path():
    m = extract_metadata(_doc("nimbus/v2/api/rest-endpoints.md"))
    assert (m.product, m.version, m.doc_type) == ("nimbus", "v2", "api_reference")


def test_generic_segments_are_skipped_for_product():
    m = extract_metadata(_doc("docs/en/relay/1.4/tutorials/quickstart.md"))
    assert m.product == "relay"
    assert m.version == "v1.4"
    assert m.doc_type == "tutorial"


def test_front_matter_overrides_and_tags():
    m = extract_metadata(_doc("x/y.md", product="Payments", version="3", tags="auth, Billing"))
    assert m.product == "payments"
    assert m.version == "v3"
    assert m.tags == ["auth", "billing"]


def test_url_sources():
    m = extract_metadata(_doc("https://docs.example.com/relay/v1/faq/troubleshooting"))
    assert m.product == "relay"
    assert m.doc_type == "faq"


def test_normalize_version():
    assert normalize_version("V2") == "v2"
    assert normalize_version("latest") == "latest"


def test_pinecone_metadata_drops_empty_values():
    meta = extract_metadata(_doc("page.md")).to_pinecone()
    assert "version" not in meta and "tags" not in meta
    assert meta["content_hash"]
