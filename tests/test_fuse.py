import numpy as np
from fuse import concat_probs, train_logreg


def test_concat_probs():
    v = np.array([0.6, 0.4])
    t = np.array([0.1, 0.2, 0.7])
    out = concat_probs(v, t)
    assert out.shape == (5,)
    assert np.allclose(out, [0.6, 0.4, 0.1, 0.2, 0.7])


def test_train_logreg():
    X = np.array([[0.6, 0.4, 0.1, 0.2, 0.7], [0.4, 0.6, 0.5, 0.3, 0.2]])
    y = np.array([0, 1])
    clf = train_logreg(X, y)
    preds = clf.predict(X)
    assert list(preds) == [0, 1]

