from collections import Counter

from pnl2vec.tokenizer import Token, TokenKind, Vocabulary


def test_reserved_ids():
    v = Vocabulary()
    assert v.token_to_id("<PAD>") == 0
    assert v.token_to_id("<UNK>") == 1
    assert v.token_to_id("<BOS>") == 2


def test_unknown_token():
    v = Vocabulary()
    assert v.token_to_id("PITCH_CLASS:Z") == v.unk_id


def test_save_load(tmp_path):
    v = Vocabulary()
    v.add("PITCH_CLASS:C", Token(TokenKind.PITCH_CLASS, "C"), frequency=3)
    path = tmp_path / "vocabulary.json"
    v.save(path)
    v2 = Vocabulary.load(path)
    assert v2.token_to_id("PITCH_CLASS:C") == v.token_to_id("PITCH_CLASS:C")
    assert v2.id_to_token(v2.token_to_id("PITCH_CLASS:C")) == "PITCH_CLASS:C"


def test_encode_decode():
    v = Vocabulary()
    v.add("DURATION:1/4", Token(TokenKind.DURATION, "1/4"), frequency=1)
    tokens = [Token(TokenKind.DURATION, "1/4")]
    ids = v.encode(tokens)
    back = v.decode(ids)
    assert back[0].canonical() == "DURATION:1/4"


def test_build_from_frequencies_no_leakage_concept():
    v = Vocabulary()
    train_freq = Counter({"PITCH_CLASS:C": 5, "DURATION:1/4": 3})
    lookup = {
        "PITCH_CLASS:C": Token(TokenKind.PITCH_CLASS, "C"),
        "DURATION:1/4": Token(TokenKind.DURATION, "1/4"),
    }
    v.build_from_frequencies(train_freq, lookup)
    assert "PITCH_CLASS:C" in v.token_to_id_
    assert v.token_to_id("PITCH_CLASS:D") == v.unk_id
