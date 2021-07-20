import pytest
import gmshnics as g4x
import numpy as np
import os


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

    return mesh, facet_f


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
    facet_arr = facet_f.array()
    bdry_idx, = np.where(facet_arr > 0)
    assert len(bdry_idx)
    
    assert set(facet_arr) == {0, 1, 2, 3, 4}

    transforms = {1: lambda y: (y - ll)[:, 0],
                  2: lambda y: (y - ur)[:, 0],
                  3: lambda y: (y - ll)[:, 1],
                  4: lambda y: (y - ur)[:, 1]}

    _, e2v = mesh.init(1, 0), mesh.topology()(1, 0)

    for tag, transform in transforms.items():
        # Get facet vertices (as index)
        indices = np.unique(np.hstack([e2v(e) for e in np.where(facet_arr == tag)[0]]))
        assert np.all(transform(x[indices]) < tol)

    return mesh, facet_f


def test_box(resolution=1.0, tol=1E-3):
    ll = np.array([1, 2, 0])
    ur = np.array([3, 4, 2])
    mesh, facet_f = g4x.gBox(ll, ur, resolution)

    assert mesh.topology().dim() == 3
    assert mesh.geometry().dim() == 3

    x = mesh.coordinates()
    # Bounds check out
    assert np.linalg.norm(np.min(x, axis=0)-ll) < 1E-3
    assert np.linalg.norm(np.max(x, axis=0)-ur) < 1E-3    

    # Taggeting checks out
    assert facet_f.dim() == 2
    facet_arr = facet_f.array()
    bdry_idx, = np.where(facet_arr > 0)
    assert len(bdry_idx)
    
    assert set(facet_arr) == {0, 1, 2, 3, 4, 5, 6}

    transforms = {1: lambda y: (y - ll)[:, 0],
                  2: lambda y: (y - ur)[:, 0],
                  3: lambda y: (y - ll)[:, 1],
                  4: lambda y: (y - ur)[:, 1],
                  5: lambda y: (y - ll)[:, 2],
                  6: lambda y: (y - ur)[:, 2]}

    _, f2v = mesh.init(2, 0), mesh.topology()(2, 0)

    for tag, transform in transforms.items():
        # Get facet vertices (as index)
        indices = np.unique(np.hstack([f2v(f) for f in np.where(facet_arr == tag)[0]]))
        assert np.all(transform(x[indices]) < tol)

    return mesh, facet_f


@pytest.mark.parametrize('f', (test_circle, test_rectangle))
def test_refine(f):
    prev = None
    for resolution in (0.1, 0.08, 0.05):
        mesh, _ = f(resolution)
        hmin = mesh.hmin()

        assert prev is None or hmin < prev
        prev = hmin

        
def test_load_dump():
    mesh, foo = test_circle(resolution=0.1, tol=1E-3)
    # With given name
    g4x.dump_h5('test.h5', mesh, {'foo_1': foo})
    mesh, entity_fs = g4x.load_h5('test.h5', entity_f_names=('foo_1', ))
    assert 'foo_1' in entity_fs
    assert entity_fs['foo_1'].dim() == 1

    # With found name
    g4x.dump_h5('test.h5', mesh, {'foo_1': foo})
    mesh, entity_fs = g4x.load_h5('test.h5', entity_f_names=None)
    assert 'foo_1' in entity_fs
    assert entity_fs['foo_1'].dim() == 1

    # With auto name
    g4x.dump_h5('test.h5', mesh, (foo, ))
    mesh, entity_fs = g4x.load_h5('test.h5', entity_f_names=None)
    assert 'entity_f_1' in entity_fs
    assert entity_fs['entity_f_1'].dim() == 1

    os.remove('test.h5')
    

def test_rectangle_aniso():
    # The idea here is that spacing on 1 should be finer than elsewhere
    ll = np.array([1, 1])
    ur = np.array([3, 5])
    mesh, facet_f = g4x.gRectangle(ll, ur, {1: {'SizeMin': 0.01, 'DistMin': 0.1, 'DistMax': 0.1, 'SizeMax': 0.1}})

    x = mesh.coordinates()
    _, e2v = mesh.init(1, 0), mesh.topology()(1, 0)

    sizes = {}
    for tag in (1, 2, 3, 4):
        # Compute mean vertex spacing on the boudary
        edges, = np.where(facet_f.array() == tag)
        v0, v1 = np.column_stack([e2v(e) for e in edges])
        hs = np.linalg.norm(x[v1] - x[v0], 2, 1)
        sizes[tag] = np.mean(hs)

    assert 1 == min(sizes, key=lambda k: sizes[k]), sizes


def test_box_aniso():
    # The idea here is that spacing on 1 should be finer than elsewhere
    ll = np.array([1, 1, 0])
    ur = np.array([3, 2, 1])
    mesh, facet_f = g4x.gBox(ll, ur, {1: {'SizeMin': 0.1, 'DistMin': 0.1, 'DistMax': 0.2, 'SizeMax': 1.}})

    x = mesh.coordinates()
    _, f2v = mesh.init(2, 0), mesh.topology()(2, 0)

    sizes = {}
    for tag in (1, 2, 3, 4, 5, 6):
        # Compute mean element size
        facets, = np.where(facet_f.array() == tag)
        v0, v1, v2 = np.column_stack([f2v(f) for f in facets])
        hs = 0.5*np.linalg.norm(np.cross(x[v1] - x[v0], x[v2] - x[v0]), 2, axis=1)
        sizes[tag] = np.mean(hs)

    assert 1 == min(sizes, key=lambda k: sizes[k]), sizes
