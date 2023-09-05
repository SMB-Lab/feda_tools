import random

from feda_tools.twodim_hist import add


def test_add():
    assert add(1, 2) == 3
    assert add(0.1, 3) == 3.1
    a, b = random.random(), random.random()
    assert add(a, b) == a + b