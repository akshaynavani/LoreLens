from pathlib import Path

from lorelens.ingestion.loaders import DirectoryLoader, html_to_markdownish, rst_to_markdownish

SAMPLE_DOCS = Path(__file__).resolve().parents[1] / "data" / "sample_docs"


def test_html_to_markdownish_keeps_structure_and_drops_chrome():
    html = """<html><head><title>Rate limits</title></head><body>
    <nav>menu</nav><main><h1>Rate limits</h1><p>600 requests per minute.</p>
    <h2>Retries</h2><pre><code>sleep(retry_after)</code></pre></main>
    <footer>copyright</footer></body></html>"""
    text, title = html_to_markdownish(html)
    assert title == "Rate limits"
    assert "# Rate limits" in text and "## Retries" in text
    assert "```\nsleep(retry_after)\n```" in text
    assert "menu" not in text and "copyright" not in text


def test_rst_headers_are_translated():
    text = rst_to_markdownish("Title\n=====\n\nBody\n\nSub\n---\n\nMore")
    assert "# Title" in text and "## Sub" in text


def test_directory_loader_reads_sample_corpus():
    docs = list(DirectoryLoader(SAMPLE_DOCS, base_url="https://docs.example.com"))
    sources = {d.source for d in docs}
    assert "nimbus/v2/guides/authentication.md" in sources
    auth = next(d for d in docs if d.source == "nimbus/v2/guides/authentication.md")
    assert auth.title == "Authentication"
    assert auth.url == "https://docs.example.com/nimbus/v2/guides/authentication"
    assert auth.metadata["tags"] == ["auth", "security"]
    assert "---" not in auth.text.splitlines()[0]  # front matter stripped
    assert len({d.doc_id for d in docs}) == len(docs)
