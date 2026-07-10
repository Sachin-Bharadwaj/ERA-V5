"""
Module 1: Corpus acquisition.

Downloads the four Wikipedia "India" articles (en, hi, te, es), strips
markup/nav/references, and writes clean UTF-8 plain-text files to data/.

Kept separate from tokenizer logic so the corpus can be refreshed
independently of everything downstream.
"""
import os
import re
import unicodedata
import requests
from bs4 import BeautifulSoup

URLS = {
    "hi": "https://hi.wikipedia.org/wiki/%E0%A4%AD%E0%A4%BE%E0%A4%B0%E0%A4%A4",
    "en": "https://en.wikipedia.org/wiki/India",
    "te": "https://te.wikipedia.org/wiki/%E0%B0%AD%E0%B0%BE%E0%B0%B0%E0%B0%A4%E0%B0%A6%E0%B1%87%E0%B0%B6%E0%B0%82",
    "es": "https://es.wikipedia.org/wiki/India",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; BPE-Tokenizer-Assignment/1.0)"}

# Elements that are not running prose and would pollute word/token statistics.
STRIP_SELECTORS = [
    "table", "sup.reference", "sup.noprint", "span.mw-editsection",
    "div.navbox", "div.vertical-navbox", "div.infobox", "div.hatnote",
    "div.thumb", "div.metadata", "div.reflist", "ol.references",
    "div.mw-references-wrap", "style", "script", "div.sistersitebox",
    "table.infobox", "div.reflist-columns", "div.shortdescription",
]


def clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("#mw-content-text .mw-parser-output")
    if content is None:
        content = soup

    for sel in STRIP_SELECTORS:
        for tag in content.select(sel):
            tag.decompose()

    # Only keep paragraph / heading prose; skip lists that are mostly links (e.g. "See also").
    parts = []
    for tag in content.find_all(["p", "h2", "h3"]):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)

    text = "\n".join(parts)
    # NFC normalization up front: fixes the "visually identical text, different
    # byte sequences" problem (decomposed combining sequences -> composed form).
    text = unicodedata.normalize("NFC", text)
    # Collapse citation brackets like [1], [23] and stray whitespace.
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def fetch_all(force: bool = False) -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    out = {}
    for lang, url in URLS.items():
        path = os.path.join(DATA_DIR, f"{lang}.txt")
        if not force and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                out[lang] = f.read()
            continue
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        text = clean_html_to_text(resp.text)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        out[lang] = text
    return out


if __name__ == "__main__":
    corpus = fetch_all(force=True)
    for lang, text in corpus.items():
        print(f"{lang}: {len(text):>8} chars, {len(text.split()):>6} whitespace-tokens")
