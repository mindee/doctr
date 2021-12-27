from doctr.transforms import modules as T


def test_imagetransform():

    transfo = T.ImageTransform(lambda x: 1 - x)
    assert transfo(0, 1) == (1, 1)


def test_samplecompose():

    transfos = [lambda x, y: (1 - x, y), lambda x, y: (x, 2 * y)]
    transfo = T.SampleCompose(transfos)
    assert transfo(0, 1) == (1, 2)


def test_oneof():
    transfos = [lambda x: 1 - x, lambda x: x + 10]
    transfo = T.OneOf(transfos)
    out = transfo(1)
    assert out == 0 or out == 11


def test_randomapply():
    transfo = T.RandomApply(lambda x: 1 - x)
    out = transfo(1)
    assert out == 0 or out == 1
    assert repr(transfo).endswith(", p=0.5)")
