import pytest
from ml.pipelines.song_identity import normalize_song_identity, canonical_song_id

def test_remaster_and_official_video_normalization():
    id_orig = canonical_song_id("Blinding Lights", "The Weeknd")
    id_remaster = canonical_song_id("Blinding Lights - Remastered 2020", "The Weeknd")
    id_video = canonical_song_id("Blinding Lights [Official Music Video]", "The Weeknd")
    id_audio = canonical_song_id("Blinding Lights (Audio)", "the weeknd")
    
    assert id_orig == id_remaster
    assert id_orig == id_video
    assert id_orig == id_audio

def test_live_version_configurable_handling():
    # When treat_live_as_same is True (default for diffusion tracking)
    id_studio = canonical_song_id("Bad Habits", "Ed Sheeran", treat_live_as_same=True)
    id_live = canonical_song_id("Bad Habits (Live)", "Ed Sheeran", treat_live_as_same=True)
    assert id_studio == id_live

    # When treat_live_as_same is False (strict recording separation)
    id_studio_strict = canonical_song_id("Bad Habits", "Ed Sheeran", treat_live_as_same=False)
    id_live_strict = canonical_song_id("Bad Habits (Live)", "Ed Sheeran", treat_live_as_same=False)
    assert id_studio_strict != id_live_strict

    # Same for traitor
    assert canonical_song_id("traitor", "Olivia Rodrigo", treat_live_as_same=True) == canonical_song_id("traitor (Live)", "Olivia Rodrigo", treat_live_as_same=True)
    assert canonical_song_id("traitor", "Olivia Rodrigo", treat_live_as_same=False) != canonical_song_id("traitor (Live)", "Olivia Rodrigo", treat_live_as_same=False)

def test_featured_artist_in_title_vs_artist_field():
    id_title_feat = canonical_song_id("Tusa (feat. Nicki Minaj)", "KAROL G")
    id_pipe_artist = canonical_song_id("Tusa", "KAROL G | Nicki Minaj")
    id_comma_artist = canonical_song_id("Tusa", "Nicki Minaj, KAROL G")
    
    assert id_title_feat == id_pipe_artist
    assert id_title_feat == id_comma_artist

def test_case_and_whitespace_insensitivity():
    id1 = canonical_song_id("  As It Was   ", " Harry Styles ")
    id2 = canonical_song_id("as it was", "harry styles")
    
    assert id1 == id2

def test_artist_ordering_invariance():
    id_order_a = canonical_song_id("STAY", "The Kid LAROI & Justin Bieber")
    id_order_b = canonical_song_id("STAY", "Justin Bieber | The Kid LAROI")
    
    assert id_order_a == id_order_b

def test_accent_and_diacritics_normalization():
    id_accent = canonical_song_id("Bellaquita (feat. Lenny Tavárez)", "Dalex")
    id_plain = canonical_song_id("Bellaquita (feat. Lenny Tavarez)", "Dalex")
    
    assert id_accent == id_plain

def test_non_latin_unicode_scripts_preserved():
    id_cyrillic = canonical_song_id("ЩО У НЕЇ В ДУМКАХ", "Амирина")
    id_japanese = canonical_song_id("本能", "Sheena Ringo")
    id_korean = canonical_song_id("나윤권", "아이유")
    
    assert id_cyrillic != id_japanese
    assert id_cyrillic != id_korean
    assert id_japanese != id_korean


def test_missing_values_do_not_crash():
    import math
    # pandas hands NaN floats for missing artists/titles; these must not raise
    assert canonical_song_id("Song", math.nan) == canonical_song_id("Song", None)
    assert isinstance(canonical_song_id(math.nan, "Artist"), str)
