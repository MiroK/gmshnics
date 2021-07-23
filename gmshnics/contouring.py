# Functionality for reasoning about nesting of contours
import numpy as np

def refine_contour(contour, n):
    '''Uniform refine n times'''
    assert n >= 0
    # Base
    if n == 0:
        return contour
    # Work horse
    if n == 1:
        # Add midpoints
        dx = np.diff(contour, axis=0)
        mids = contour[:-1] + 0.5*dx
        # Combine orig, new and close
        return np.row_stack(list(zip(contour[:-1], mids)) + [contour[0]])
    # Iter
    return refine_contour(refine_contour(contour, 1), n-1)


def is_inside_contour_it(contour, points, tol=1E-2, nrefs=0):
    '''We say close to 0 is outside'''
    for wn in wind_number_it(contour, points, nrefs):
        yield abs(wn) > tol

        
def is_inside_contour(contour, points, tol=1E-2, nrefs=0):
    '''We say close to 0 is outside'''
    return np.fromiter(is_inside_contour_it(contour, points, tol=tol, nrefs=nrefs),
                       dtype=bool)


def wind_number_it(contour, points, nrefs=0):
    '''If contour is defined by linked vertices'''
    # Improve accuracy of integral by refinement
    if nrefs > 0:
        contour = refine_contour(contour, nrefs)
    # Look at contour integral of dot(n, x-c/dot(x-c, x-c))
    # We assume closed contour so
    assert np.linalg.norm(contour[0] - contour[-1]) < 1E-13
    assert contour.ndim == 2
    # Precondpute edge length
    dx, dy = np.diff(contour, axis=0).T
    dl = np.sqrt(dx**2 + dy**2)
    # Unit normal
    n = np.c_[dy, -dx]/dl.reshape((-1, 1))
    # And midpoint
    x = contour[:-1] + 0.5*np.c_[dx, dy]
    # NOTE: here we do the integral numerically so we are limited by
    # the approximation; it's also mid point rule type quadrature since
    # that is where the normal makes sense (cf. vertices)
    for c in points:
        yield (1./2./np.pi)*np.sum(dl*np.sum((x-c)*n, axis=1)/np.linalg.norm(x-c, 2, axis=1)**2)


def wind_number(contour, points, nrefs=0):
    '''If contour is defined by linked vertices'''
    return np.fromiter(wn_number(contour, points), dtype=float)


def bounding_box(contour):
    '''Extrema of axis aligned coordinates'''
    return (np.min(contour, axis=0), np.max(contour, axis=0))


def encloses_bbox(b0, b1, tol=1E-10):
    '''Does bbox0 encloses bbox1'''
    ll0, ur0 = b0
    ll1, ur1 = b1

    return np.all(ll0-tol < ll1) and np.all(ur1 < ur0+tol)


def encloses_contour(c0, c1, tol=1E-2, nrefs=0):
    '''c0 encloses c1 if all c1 points are inside c0'''
    # Bbox <= is necessary
    if not encloses_bbox(bounding_box(c0), bounding_box(c1)):
        return False
    # Hard work ...
    return all(is_inside_contour_it(c0, c1, tol=tol, nrefs=nrefs))
