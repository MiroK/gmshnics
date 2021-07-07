import pytest
import gmshnics as g4x
import numpy as np


def test_circle(resolution=0.1, tol=1E-3):
    center = np.array([1, 2])
    radius = 0.8
    mesh, facet_f = g4x.gCircle(center, radius, resolution)

    assert mesh.topology().dim() == 2
    assert mesh.geometry().dim() == 2

    x = mesh.coordinates()
    # The center checks out
    assert np.linalg.norm(np.mean(x, axis=0)-center) < 1E-3
    # The radius checks out
    assert np.all(np.linalg.norm(x-center, 2, axis=1)**2 < radius**2+tol)

    # Taggeting checks out
    assert facet_f.dim() == 1
    bdry_idx, = np.where(facet_f.array() == 1)
    assert len(bdry_idx)

    _, e2v = mesh.init(1, 0), mesh.topology()(1, 0)
    for bdry_edge in bdry_idx:
        x0, x1 = x[e2v(bdry_edge)]
        assert radius-tol < np.linalg.norm(x0 - center, 2) < radius+tol
        assert radius-tol < np.linalg.norm(x1 - center, 2) < radius+tol

    return mesh


def test_rectangle(resolution=0.1, tol=1E-3):
    ll = np.array([1, 2])
    ur = np.array([3, 4])
    mesh, facet_f = g4x.gRectangle(ll, ur, resolution)

    assert mesh.topology().dim() == 2
    assert mesh.geometry().dim() == 2

    x = mesh.coordinates()
    # Bounds check out
    assert np.linalg.norm(np.min(x, axis=0)-ll) < 1E-3
    assert np.linalg.norm(np.max(x, axis=0)-ur) < 1E-3    

    # Taggeting checks out
    assert facet_f.dim() == 1
    facet_f = facet_f.array()
    bdry_idx, = np.where(facet_f > 0)
    assert len(bdry_idx)
    
    assert set(facet_f) == {0, 1, 2, 3, 4}

    transforms = {1: lambda y: (y - ll)[:, 0],
                  2: lambda y: (y - ur)[:, 0],
                  3: lambda y: (y - ll)[:, 1],
                  4: lambda y: (y - ur)[:, 1]}

    _, e2v = mesh.init(1, 0), mesh.topology()(1, 0)

    for tag, transform in transforms.items():
        # Get facet vertices (as index)
        indices = np.unique(np.hstack([e2v(e) for e in np.where(facet_f == tag)[0]]))
        assert np.all(transform(x[indices]) < tol)

    return mesh


@pytest.mark.parametrize('f', (test_circle, ))
def test_refine(f):
    prev = None
    for resolution in (0.1, 0.08, 0.05):
        mesh = f(resolution)
        hmin = mesh.hmin()

        assert prev is None or hmin < prev
        prev = hmin
