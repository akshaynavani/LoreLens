from lorelens.ingestion.chunking import (
    CodeAwareChunker,
    MarkdownChunker,
    TokenChunker,
    count_tokens,
    get_chunker,
    split_sections,
)
from lorelens.ingestion.metadata import extract_metadata
from lorelens.models import SourceDocument

DOC = """# Authentication

Intro paragraph about authentication in the product.

## Tokens

Tokens expire after 3600 seconds. Request a new one when you receive a 401.

```bash
curl -X POST https://auth.example.com/token -d grant_type=client_credentials
```

## Scopes

Use storage:read and storage:write scopes for object access.
"""


def _doc(text: str = DOC) -> SourceDocument:
    return SourceDocument(doc_id="abc", text=text, source="nimbus/v2/guides/auth.md", title="Auth")


def test_split_sections_builds_breadcrumbs():
    sections = split_sections(DOC)
    crumbs = [s.breadcrumb for s in sections]
    assert "Authentication > Tokens" in crumbs
    assert "Authentication > Scopes" in crumbs


def test_markdown_chunker_ids_and_sections():
    doc = _doc()
    chunks = MarkdownChunker(256, 32).split(doc, extract_metadata(doc))
    assert [c.chunk_id for c in chunks] == [f"abc#{i}" for i in range(len(chunks))]
    token_chunk = next(c for c in chunks if c.section == "Authentication > Tokens")
    assert "3600 seconds" in token_chunk.text
    assert token_chunk.embedding_text().startswith("Auth | Authentication > Tokens")


def test_code_aware_never_splits_small_code_block():
    filler = "\n\n".join(f"Paragraph {i} " + "words " * 40 for i in range(6))
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(30)) + "\n```"
    text = f"# Guide\n\n{filler}\n\n{code}\n\n{filler}"
    doc = _doc(text)
    chunks = CodeAwareChunker(200, 20).split(doc, extract_metadata(doc))
    with_code = [c for c in chunks if "x0 = 0" in c.text]
    assert len(with_code) == 1
    assert "x29 = 29" in with_code[0].text
    assert with_code[0].text.count("```") % 2 == 0


def test_code_aware_splits_oversized_code_block_and_refences():
    code = "```python\n" + "\n".join(f"value_{i} = compute({i})" for i in range(400)) + "\n```"
    doc = _doc(f"# Big\n\n{code}")
    chunks = CodeAwareChunker(128, 16).split(doc, extract_metadata(doc))
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.count("```") == 2
        assert c.token_count <= 128 + 16


def test_token_chunker_respects_budget():
    doc = _doc("word " * 3000)
    chunks = TokenChunker(100, 10).split(doc, extract_metadata(doc))
    assert len(chunks) > 10
    assert all(count_tokens(c.text) <= 100 for c in chunks)


def test_unknown_strategy_raises():
    import pytest

    with pytest.raises(ValueError):
        get_chunker("semantic-magic", 100, 10)
