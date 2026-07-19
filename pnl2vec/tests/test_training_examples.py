import random

from pnl2vec.corpus import load_corpus, split_documents
from pnl2vec.pnl import parse_pnl
from pnl2vec.tokenizer import Tokenizer, Vocabulary
from pnl2vec.training.dataset import ContextConfig, ContextPolicy, annotated_to_instances, generate_pairs_for_document
from pnl2vec.training.trainer import build_vocabulary


def test_pairs_do_not_cross_documents(examples_dir):
    docs = load_corpus(examples_dir)
    assert len(docs) >= 2
    tok = Tokenizer()
    vocab = build_vocabulary(docs, tok)
    cfg = ContextConfig(policy=ContextPolicy.HYBRID)
    rng = random.Random(0)
    all_pairs = []
    for d in docs:
        ann = tok.tokenize_annotated(d.document)
        inst = annotated_to_instances(ann, vocab)
        pairs = generate_pairs_for_document(inst, doc_id=d.doc_id, config=cfg, rng=rng)
        assert all(p.doc_id == d.doc_id for p in pairs)
        all_pairs.extend(pairs)
    assert all_pairs


def test_simultaneous_pairs(examples_dir):
    docs = [d for d in load_corpus(examples_dir) if d.doc_id == "polyphonic"]
    if not docs:
        return
    tok = Tokenizer()
    vocab = build_vocabulary(docs, tok)
    ann = tok.tokenize_annotated(docs[0].document)
    inst = annotated_to_instances(ann, vocab)
    pairs = generate_pairs_for_document(
        inst,
        doc_id="polyphonic",
        config=ContextConfig(policy=ContextPolicy.TEMPORAL),
        rng=random.Random(0),
    )
    # May be empty if offsets missing; at least should not crash
    assert isinstance(pairs, list)


def test_split_by_document(examples_dir):
    docs = load_corpus(examples_dir)
    split = split_documents(docs, seed=42)
    train_ids = {d.doc_id for d in split.train}
    test_ids = {d.doc_id for d in split.test}
    assert train_ids.isdisjoint(test_ids)
