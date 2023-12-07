import re

import contractions
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

STOP_WORD_LIST = [
    "a",
    "an",
    "a",
    "the",
    "and",
    "at",
    "by",
    "to",
    "in",
    "out",
    "y",
    "are",
    "is",
    "as",
    "t",
    "of",
    "for",
]


def remove_http_links(text):
    return re.sub(r"http\S+|www.\S+", "", text)


def remove_html_tags(text):
    return re.sub(r"<[^>]*>", "", text)


def replace_digits(match):
    char_map = {"1": "i", "0": "o", "3": "e"}
    return char_map[match.group(0)]


def replace_digit_with_letter(text):
    """
    Transforms specified digits within a given string 'text' into their respective letter counterparts
    commonly used in leetspeak or similar stylizations.

    This function targets the digits '1', '0', and '3' only when they are sandwiched between
    alphabetic characters, and replaces them with 'i', 'o', and 'e', respectively.
    """
    return re.sub(r"(?<=[a-zA-Z])[103](?=[a-zA-Z])", replace_digits, text)


def reduce_repeated_letters(text):
    """
    Compresses sequences of identical letters occurring more than twice in a row within a given string 'text'
    to a single instance of that letter.

    This function is particularly useful in normalizing text with exaggerated letter repetitions, commonly found in
    informal communication like social media posts or text messages.

    Example:
        heeeello -> hello
        worlllld -> world
    """
    return re.sub(r"(.)\1{2,}", r"\1", text)


def reduce_haha(text):
    return re.sub(r"\bhaha\w*\b", "haha", text)


def replace_hyphens_with_spaces(text):
    """
    Replaces all hyphens '-' in the string with spaces ' '.
    """
    return text.replace("-", " ")


def replace_newlines_and_tabs_with_space(text):
    return re.sub(r"[\n\t]", " ", text)


def remove_non_word_characters(text):
    """
    Removes all characters from the string that are not alphanumeric (letters and numbers) or whitespace.
    """
    return re.sub(r"[^\w\s]", "", text)


def replace_prefix_or_word(prefix_to_replace, new_prefix, text):
    """
    Replaces occurrences of a specified prefix with a new prefix in the given text.
    This function targets:
    - The prefix when it is a standalone word.
    - The prefix when it appears at the start of other words.
    """
    # Pattern to match the prefix when it's a standalone word or at the start of other words
    pattern = r"\b{}(\b|\w)".format(re.escape(prefix_to_replace))
    return re.sub(
        pattern,
        lambda m: new_prefix + (m.group(1) if m.group(1).isalnum() else ""),
        text,
    )


def remove_spaces_from_spaced_words(text):
    pattern = r"(?:\b\w\s){2,}\w\b"

    def replace_spaces(match):
        return match.group().replace(" ", "")

    return re.sub(pattern, replace_spaces, text)


def remove_digits(text):
    """
    Removes all digits from the string.
    """
    return re.sub(r"\d+", "", text)


def normalize_whitespace(text):
    """
    Normalizes whitespace in the string, replacing multiple consecutive whitespace characters with a single space,
    and trims leading and trailing whitespace.
    """
    return re.sub(r"\s+", " ", text).strip()


def filter_out_stop_words(tokens, stop_words_list):
    """
    Filters out stop words from a list of tokens.
    """
    return [word for word in tokens if word not in stop_words_list]


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize_word(word, pos_tag):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word, pos=pos_tag)


def lemmatize_tokens(tokens):
    """
    Performs lemmatization on a list of word tokens.
    """
    pos_tags = nltk.pos_tag(tokens)
    return [lemmatize_word(word, get_wordnet_pos(pos)) for word, pos in pos_tags]


def filter_text(text, with_lematization=False):
    text = text.lower()
    text = remove_html_tags(text)
    text = remove_http_links(text)
    text = replace_digit_with_letter(text)
    text = reduce_repeated_letters(text)
    text = reduce_haha(text)
    text = replace_hyphens_with_spaces(text)
    text = contractions.fix(text)
    text = remove_non_word_characters(text)
    text = replace_prefix_or_word("fck", "fuck", text)
    text = remove_digits(text)
    text = remove_spaces_from_spaced_words(text)
    text = normalize_whitespace(text)

    tokens = text.split()

    if with_lematization:
        tokens = lemmatize_tokens(tokens)

    tokens = filter_out_stop_words(tokens, STOP_WORD_LIST)

    return " ".join(tokens)
