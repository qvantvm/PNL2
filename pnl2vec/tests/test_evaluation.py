import numpy as np

from pnl2vec.api import PNL2Vec
from pnl2vec.evaluation.similarity import EmbeddingIndex
from pnl2vec.tokenizer import Token, TokenKind, Tokenizer, Vocabulary


def _model():
    v = Vocabulary()
    for name, kind, val in [
        ("PITCH_CLASS:C", TokenKind.PITCH_CLASS, "C"),
        ("PITCH_CLASS:D", TokenKind.PITCH_CLASS, "D"),
        ("DURATION:1/4", TokenKind.DURATION, "1/4"),
        ("OCTAVE:4", TokenKind.OCTAVE, 4),
    ]:
        v.add(name, Token(kind, val), frequency=5)
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(v), 16)).astype(np.float64)
    # Make C and D somewhat similar
    emb[v.token_to_id("PITCH_CLASS:D")] = emb[v.token_to_id("PITCH_CLASS:C")] + 0.01
    return PNL2Vec(v, emb, Tokenizer())


def test_nearest_neighbor_self_exclusion():
    model = _model()
    neigh = model.nearest_neighbors("PITCH_CLASS:C", top_k=5)
    assert all(n.token != "PITCH_CLASS:C" for n in neigh)
    assert neigh[0].token == "PITCH_CLASS:D"


def test_phrase_pooling(tiny_scale_text):
    model = _model()
    # encode may map many tokens to UNK; still should return vector
    vec = model.embed_pnl(tiny_scale_text)
    assert vec.shape == (16,)
    assert np.isfinite(vec).all()


def test_similarity_symmetric():
    model = _model()
    a = model.similarity("PITCH_CLASS:C", "PITCH_CLASS:D")
    b = model.similarity("PITCH_CLASS:D", "PITCH_CLASS:C")
    assert abs(a - b) < 1e-6
