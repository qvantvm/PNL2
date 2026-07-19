from typer.testing import CliRunner

from pnl2vec.cli import app
from pnl2vec.corpus import generate_corpus

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.stdout


def test_cli_validate_example(examples_dir):
    result = runner.invoke(app, ["validate", str(examples_dir / "tiny_scale.pnl")])
    assert result.exit_code == 0
    assert "OK" in result.stdout


def test_cli_generate_synthetic(tmp_path):
    out = tmp_path / "raw"
    # Use API directly for speed/determinism; CLI smoke via invoke
    paths = generate_corpus("tiny", seed=1, output_dir=out, force=True)
    assert len(paths) == 100
    # spot-check a few parse
    from pnl2vec.pnl import parse_pnl

    for p in paths[:5]:
        parse_pnl(p.read_text(encoding="utf-8"))


def test_deterministic_synthetic(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_corpus("tiny", seed=7, output_dir=a, force=True)
    generate_corpus("tiny", seed=7, output_dir=b, force=True)
    texts_a = [p.read_text() for p in sorted(a.glob("*.pnl"))]
    texts_b = [p.read_text() for p in sorted(b.glob("*.pnl"))]
    assert texts_a == texts_b
