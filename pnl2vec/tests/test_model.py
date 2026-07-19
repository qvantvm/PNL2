import torch

from pnl2vec.models import CBOWNS, NegativeSampler, SkipGramNS
from pnl2vec.tokenizer import Token, TokenKind, Vocabulary


def _tiny_vocab():
    v = Vocabulary()
    for i, name in enumerate(["PITCH_CLASS:C", "PITCH_CLASS:D", "DURATION:1/4", "OCTAVE:4"]):
        kind = TokenKind.PITCH_CLASS if name.startswith("PITCH") else (
            TokenKind.DURATION if name.startswith("DUR") else TokenKind.OCTAVE
        )
        val = name.split(":", 1)[1]
        v.add(name, Token(kind, int(val) if kind == TokenKind.OCTAVE else val), frequency=10 - i)
    return v


def test_skipgram_shapes_and_loss():
    v = _tiny_vocab()
    model = SkipGramNS(len(v), 16)
    centers = torch.tensor([v.token_to_id("PITCH_CLASS:C")] * 4)
    contexts = torch.tensor([v.token_to_id("OCTAVE:4")] * 4)
    negs = torch.randint(0, len(v), (4, 5))
    loss = model(centers, contexts, negs)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_cbow_shapes():
    v = _tiny_vocab()
    model = CBOWNS(len(v), 16)
    ctx = torch.tensor([[v.token_to_id("PITCH_CLASS:C"), v.token_to_id("OCTAVE:4")]] * 3)
    centers = torch.tensor([v.token_to_id("DURATION:1/4")] * 3)
    negs = torch.randint(0, len(v), (3, 4))
    loss = model(ctx, centers, negs)
    assert torch.isfinite(loss)


def test_negative_sampler_excludes_specials():
    v = _tiny_vocab()
    sampler = NegativeSampler.from_vocabulary(v, seed=0)
    samples = sampler.sample(20, batch_size=5).reshape(-1).tolist()
    assert v.pad_id not in samples


def test_loss_decreases_on_toy():
    v = _tiny_vocab()
    model = SkipGramNS(len(v), 32)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    centers = torch.tensor([v.token_to_id("PITCH_CLASS:C")] * 32)
    contexts = torch.tensor([v.token_to_id("OCTAVE:4")] * 32)
    negs = torch.randint(2, len(v), (32, 5))
    losses = []
    for _ in range(25):
        opt.zero_grad()
        loss = model(centers, contexts, negs)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0]


def test_checkpoint_roundtrip(tmp_path):
    from pnl2vec.training.checkpoint import load_checkpoint, save_checkpoint

    v = _tiny_vocab()
    model = SkipGramNS(len(v), 8)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=opt, epoch=1, metrics={"loss": 1.0})
    ckpt = load_checkpoint(path)
    model2 = SkipGramNS(len(v), 8)
    model2.load_state_dict(ckpt["model_state"])
    for a, b in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(a, b)
