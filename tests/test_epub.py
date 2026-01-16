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
