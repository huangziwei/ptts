from ptts.epub import html_to_text


def test_html_to_text_dropcap() -> None:
    html = b"<html><body><p><span class='dropcap'>T</span>his book.</p></body></html>"
    assert html_to_text(html) == "This book."


def test_html_to_text_removes_footnotes() -> None:
    html = (
        b"<html><body>"
        b"<p>Sentence<sup><a epub:type='noteref'>1</a></sup>.</p>"
        b"<aside epub:type='footnote'>Footnote text.</aside>"
        b"<p class='footnote'>Another footnote.</p>"
        b"</body></html>"
    )
    assert html_to_text(html) == "Sentence."


def test_html_to_text_normalizes_modifier_apostrophe() -> None:
    html = "<html><body><p>It\u02bcs fine.</p></body></html>".encode("utf-8")
    assert html_to_text(html) == "It's fine."


def test_html_to_text_normalizes_curly_apostrophe() -> None:
    html = "<html><body><p>It\u2019s fine.</p></body></html>".encode("utf-8")
    assert html_to_text(html) == "It's fine."
