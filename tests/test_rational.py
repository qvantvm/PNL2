from pnl2.rational import Rational, effective_duration


def test_reduce():
    assert str(Rational(2, 8)) == "1/4"
    assert str(Rational(3, 8)) == "3/8"


def test_augment():
    assert effective_duration("1/4", 1) == Rational(3, 8)
    assert effective_duration("1/4", 2) == Rational(7, 16)


def test_tuplet():
    assert effective_duration("1/8", 0, (3, 2)) == Rational(1, 12)
