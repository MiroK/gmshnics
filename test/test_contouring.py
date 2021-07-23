import gmshnics.contouring as contour
import numpy as np


def test_contour():
    # L shaped one
    Lc = np.array([[0.5, 0.5],
                   [0.5, 0.0],
                   [1.0, 0.0],
                   [1.0, 1.0],
                   [0.0, 1.0],
                   [0.0, 0.5],
                   [0.5, 0.5]])

    points = np.array([[0.75, 0.75], [-1, -1], [0.234, 0.769]])
    mine = contour.is_inside_contour(contour=Lc, points=points, tol=1E-2, nrefs=3).tolist()
    assert mine == [True, False, True]

    # Square the is outside of L
    S = np.array([[0.0, 0.0],
                  [-1.0, 0.0],
                  [-1.0, -1.0],
                  [0.0, -1.0],
                  [0.0, 0.0]])

    assert contour.encloses_bbox(contour.bounding_box(Lc),
                                 contour.bounding_box(S)) == False

    assert contour.encloses_contour(Lc, S, nrefs=2) == False

    # D the one which bbox would not get
    D = np.array([[0.125, 0.125],
                  [0.475, 0.125],
                  [0.475, 0.475],
                  [0.125, 0.475],
                  [0.125, 0.125]])

    assert contour.encloses_bbox(contour.bounding_box(Lc),
                                 contour.bounding_box(D)) == True

    assert contour.encloses_contour(Lc, D, nrefs=2) == False

    # Thing that encloses all of them
    B = np.array([[2.0, 2.0],
                  [-2.0, 2.0],
                  [-2.0, -2.0],
                  [2.0, -2.0],
                  [2.0, 2.0]])

    assert contour.encloses_bbox(contour.bounding_box(B),
                                 contour.bounding_box(S)) == True

    assert contour.encloses_bbox(contour.bounding_box(B),
                                 contour.bounding_box(Lc)) == True
    
    assert contour.encloses_contour(B, D, nrefs=2) == True
    assert contour.encloses_contour(B, S, nrefs=2) == True
    assert contour.encloses_contour(B, Lc, nrefs=2) == True 
