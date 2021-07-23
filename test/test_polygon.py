from gmshnics.shapes import Polygon
import numpy as np
import gmsh


def test_rectangle():
    vertices = np.array([[0, 0],
                         [2, 0],
                         [2, 1],
                         [0, 1]])
    p = Polygon(vertices)
    assert np.linalg.norm(p.com - np.array([1, 0.5])) < 1E-10

    points = 3*np.random.rand(20, 2)
    x, y = points.T
    tol = 1E-3

    inside0 = np.logical_and(np.logical_and(x > tol, x < 2-tol),
                             np.logical_and(y > tol, y < 1-tol))
    inside = np.array([p.is_inside(x) for x in points])
    assert np.all(inside == inside0)

    
def test_Lshape():
    vertices = np.array([[0, 0],
                         [1, 0],
                         [1, 1],
                         [2, 1],
                         [2, 2],
                         [0, 1]])

    p = Polygon(vertices, nrefs=0)
    # Trust gmsh for ground truth
    gmsh.initialize()
    vol = p.add(gmsh.model, gmsh.model.occ)
    com = gmsh.model.occ.getCenterOfMass(*vol)[:2]
    assert np.linalg.norm(p.com - com) < 1E-10

    points = 3*np.random.rand(10, 2)
    print(points)

    def inside_rect(p, ll, ur, tol=1E-8):
        x, y = p.T
        return np.logical_and(np.logical_and(x > ll[0]-tol, x < ur[0]+tol),
                              np.logical_and(y > ll[1]-tol, y < ur[1]+tol))
    
    inside0 = np.logical_or(
        inside_rect(points, (0, 0), (1, 1)),  inside_rect(points, (0, 1), (2, 2))
    )

    inside = np.array([p.is_inside(x) for x in points])
    assert sum(inside == inside0) == len(points), (inside, inside0)

test_Lshape()
