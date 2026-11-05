import re
import unicodedata
import hashlib
from typing import Tuple, List, Set, Optional

def remove_combining_diacritics(text: str) -> str:
    """Removes combining diacritical marks (e.g. accents) while preserving non-Latin alphabets."""
    if not isinstance(text, str) or not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")

def extract_features_and_clean_title(title: Optional[str], treat_live_as_same: bool = True) -> Tuple[str, List[str]]:
    """
    Cleans a song title by:
    - Stripping bracketed video/metadata tags (e.g. [Official Video], [Audio])
    - Stripping parenthetical version/remaster tags (e.g. (Remastered 2020), (Lyric Video))
    - Extracting featured artists from title (e.g. 'Song (feat. Artist B)')
    - Standardizing punctuation and spacing while preserving non-Latin scripts (Cyrillic, CJK, etc.).
    - If treat_live_as_same is False, retains live tags to keep live versions distinct.
    """
    if not title or not isinstance(title, str):
        return "", []
    
    # Normalize unicode and strip accents/diacritics without dropping non-Latin scripts
    t = remove_combining_diacritics(title).lower()
    
    # Remove bracketed metadata [Official Video], [Audio], [HD], etc.
    t = re.sub(r"\[.*?\]", " ", t)
    
    # Base parenthetical tags to strip
    if treat_live_as_same:
        t = re.sub(r"\((?:official|music video|video|audio|remastered|remaster|lyric|lyrics|deluxe|version|live|hd|4k).*?\)", " ", t)
        t = re.sub(r"-(?:\s*remastered|\s*remaster|\s*live|\s*radio edit|\s*edit).*?$", " ", t)
    else:
        # Keep 'live' distinct
        t = re.sub(r"\((?:official|music video|video|audio|remastered|remaster|lyric|lyrics|deluxe|version|hd|4k).*?\)", " ", t)
        t = re.sub(r"-(?:\s*remastered|\s*remaster|\s*radio edit|\s*edit).*?$", " ", t)
    
    # Extract featured artists from title
    feat_artists: List[str] = []
    feat_match = re.search(r"[\(\[]?\s*(?:feat|ft|featuring)\.?\s+([^\)\]]+)[\)\]]?", t, re.IGNORECASE)
    if feat_match:
        feat_str = feat_match.group(1)
        feat_artists = [x.strip() for x in re.split(r"\||,|\band\b|&", feat_str) if x.strip()]
        t = t[:feat_match.start()] + " " + t[feat_match.end():]
        
    # Clean punctuation and excess whitespace (keeps all Unicode alphanumeric characters)
    t = re.sub(r"[^\w\s]", " ", t)
    clean_title = re.sub(r"\s+", " ", t).strip()
    return clean_title, feat_artists

def normalize_song_identity(title: Optional[str], artists: Optional[str], treat_live_as_same: bool = True) -> Tuple[str, str, str]:
    """
    Normalizes a song's title and artist list into a deterministic canonical song ID,
    clean title, and sorted artist string.
    """
    clean_title, feat_from_title = extract_features_and_clean_title(title, treat_live_as_same=treat_live_as_same)
    
    a_raw = remove_combining_diacritics(artists if isinstance(artists, str) else "").lower()
    artist_splits = re.split(r"\||,|\band\b|&|\bfeat\.?\b|\bft\.?\b|\bfeaturing\b", a_raw)
    
    all_artists = artist_splits + feat_from_title
    clean_tokens: Set[str] = set()
    for tok in all_artists:
        tok_clean = re.sub(r"[^\w\s]", " ", tok)
        tok_clean = re.sub(r"\s+", " ", tok_clean).strip()
        if tok_clean:
            clean_tokens.add(tok_clean)
            
    norm_artists = " ".join(sorted(clean_tokens))
    key_str = f"{norm_artists}:::{clean_title}"
    canonical_id = hashlib.sha1(key_str.encode("utf-8")).hexdigest()[:16]
    return canonical_id, clean_title, norm_artists

def canonical_song_id(title: Optional[str], artists: Optional[str], treat_live_as_same: bool = True) -> str:
    """Convenience wrapper returning only the 16-character canonical song ID."""
    return normalize_song_identity(title, artists, treat_live_as_same=treat_live_as_same)[0]
