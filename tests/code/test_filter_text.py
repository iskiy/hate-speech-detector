import pytest

from src.filter_text import *


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("Check this link: http://example.com", "Check this link: "),
        ("Check this link: https://example.com", "Check this link: "),
        (
            "Check http://example.com this link: https://example.com",
            "Check  this link: ",
        ),
        ("No link here", "No link here"),
    ],
)
def test_remove_http_links(input_text, expected):
    assert remove_http_links(input_text) == expected


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("<p>Hello World!</p>", "Hello World!"),
        ("No tags here", "No tags here"),
        ("<a href='http://example.com'>site</a>", "site"),
        ("<tagtag>hello</tagtag>", "hello"),
    ],
)
def test_remove_html_tags(input_text, expected):
    assert remove_html_tags(input_text) == expected


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("h3llo w0rld", "hello world"),
        ("12345", "12345"),
        ("123", "123"),
        ("103", "103"),
        ("t3st", "test"),
        ("a1a", "aia"),
        ("No digits", "No digits"),
    ],
)
def test_replace_digit_with_letter(input_text, expected):
    assert replace_digit_with_letter(input_text) == expected


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("soooo longgg", "so long"),
        ("normal", "normal"),
        ("eeeeeek", "ek"),
        ("heeeeeeeeeeeelloooooooooo", "hello"),
    ],
)
def test_reduce_repeated_letters(input_text, expected):
    assert reduce_repeated_letters(input_text) == expected


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("hahahahaha", "haha"),
        ("hahah", "haha"),
        ("haha okay haha", "haha okay haha"),
        ("No haha here", "No haha here"),
    ],
)
def test_reduce_haha(input_text, expected):
    assert reduce_haha(input_text) == expected


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("well-done", "well done"),
        ("high-quality", "high quality"),
        ("no-hyphens", "no hyphens"),
    ],
)
def test_replace_hyphens_with_spaces(input_text, expected):
    assert replace_hyphens_with_spaces(input_text) == expected


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Line1\nLine2", "Line1 Line2"),
        ("Tab\tSeparated", "Tab Separated"),
        ("No newline or tab", "No newline or tab"),
        ("\nStarts with newline", " Starts with newline"),
        ("Ends with tab\t", "Ends with tab "),
        ("\tStarts with tab\nEnds with newline", " Starts with tab Ends with newline"),
        ("\nMultiple\nNew\nLines\n", " Multiple New Lines "),
        ("\tMultiple\tTabs\t", " Multiple Tabs "),
        ("\nMixed\nNewlines\tand\tTabs\t", " Mixed Newlines and Tabs "),
    ],
)
def test_replace_newlines_and_tabs_with_space(input_text, expected_output):
    assert replace_newlines_and_tabs_with_space(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("Hello! #World", "Hello World"),
        ("JustText", "JustText"),
        ("[Brackets], {braces}!", "Brackets braces"),
    ],
)
def test_remove_non_word_characters(input_text, expected):
    assert remove_non_word_characters(input_text) == expected


@pytest.mark.parametrize(
    "prefix_to_replace, new_prefix, input_text, expected",
    [
        ("pre", "post", "prefix and pretest", "postfix and posttest"),
        ("un", "re", "unhappy uncool un", "rehappy recool re"),
    ],
)
def test_replace_prefix_or_word(prefix_to_replace, new_prefix, input_text, expected):
    assert replace_prefix_or_word(prefix_to_replace, new_prefix, input_text) == expected


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("a b c", "abc"),
        ("normal text", "normal text"),
        ("s p a c e", "space"),
        ("hello w o r l d hello", "hello world hello"),
    ],
)
def test_remove_spaces_from_spaced_words(input_text, expected):
    assert remove_spaces_from_spaced_words(input_text) == expected


@pytest.mark.parametrize(
    "input_text, expected",
    [("abc123", "abc"), ("no digits", "no digits"), ("12345", "")],
)
def test_remove_digits(input_text, expected):
    assert remove_digits(input_text) == expected


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("  too   many spaces   ", "too many spaces"),
        ("normal", "normal"),
    ],
)
def test_normalize_whitespace(input_text, expected):
    assert normalize_whitespace(input_text) == expected


@pytest.mark.parametrize(
    "tokens, expected_output",
    [
        (["this", "is", "a", "test"], ["this", "test"]),
        (
            ["with", "some", "additional", "words"],
            ["with", "some", "additional", "words"],
        ),
        (
            ["an", "example", "with", "stop", "words"],
            ["example", "with", "stop", "words"],
        ),
        (["no", "stopwords", "here"], ["no", "stopwords", "here"]),
    ],
)
def test_filter_out_stop_words(tokens, expected_output):
    assert filter_out_stop_words(tokens, STOP_WORD_LIST) == expected_output


@pytest.mark.parametrize(
    "tokens, expected_output",
    [
        (["running", "cats"], ["run", "cat"]),
        (["bigger", "houses"], ["big", "house"]),
        (["happily", "eating"], ["happily", "eat"]),
    ],
)
def test_lemmatize_tokens(tokens, expected_output):
    assert lemmatize_tokens(tokens) == expected_output


@pytest.mark.parametrize(
    "input_text, with_lemmatization, expected_output",
    [
        (
            "Check http://example.com <b>Some</b> t3xt! It's a test.",
            False,
            "check some text it test",
        ),
        ("Hahahahaha, well-done!!!! fck", False, "haha well done fuck"),
        ("a b 123 c -- too   many   ", False, "b c too many"),
        ("Running runners ran", True, "run runner run"),
        (
            "This is-a test, for fck's <i>sake</i>! 123 hahaha",
            False,
            "this test fucks sake haha",
        ),
        (
            "Visit my <a href='http://example.com'>site</a>. It's great for 1-stop shopping & fck spam!",
            False,
            "visit my site it great stop shopping fuck spam",
        ),
        (
            "2020 - a y3ar of cha0s, hahahaha, and a b c d e f g!",
            True,
            "year chaos haha abcdefg",
        ),
        (
            "Running in 3-2-1! The quick-brown foxes were jumping over 12 lazy d0gs, hahahahaha!",
            True,
            "run quick brown fox be jump over lazy dog haha",
        ),
        (
            "I ❤️ running 100m in 10s! It's awesome😀 #Winner",
            False,
            "i running m s it awesome winner",
        ),
    ],
)
def test_filter_text(input_text, with_lemmatization, expected_output):
    assert filter_text(input_text, with_lemmatization) == expected_output
