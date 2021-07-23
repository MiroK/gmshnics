from gmshnics.shapes import Circle, Rectangle
from gmshnics.tagging import tag_volume_bdry
from gmshnics.interopt import occ
import numpy as np
import gmsh


@occ(2, 1)
def gCircle(center, radius, size, model, factory, view=False):
    '''
    Centered at center with radius. Returns mesh and FacetFunction 
    with outer boundary set to 1
    '''
    circle = Circle(center, radius)
    tdim, vol = circle.add(model, factory)
    
    factory.synchronize()

    model.addPhysicalGroup(tdim, [vol], tag=1)

    tag_volume_bdry(circle, [(tdim, vol)], model, factory)

    return size


def gUnitCircle(size):
    '''(0, 0) with 1'''
    return g.Circle(np.array([0, 0]), 1, size)


@occ(2, 1)
def gRectangle(ll, ur, size, model, factory, view=False):
    '''
    Rectangle marked by lower left and upper right corners. The returned
    facet funtion is such that
      4
    1   2
      3
    If size is dict then we expact a mapping from tag to specs of Threshold
    field in gmsh, i.e. SizeMax for size if thresholded > DistMax and 
    analogously for SizeMin and DistMin
    '''
    rect = Rectangle(ll, ur)
    tdim, vol = rect.add(model, factory)
    
    factory.synchronize()

    model.addPhysicalGroup(tdim, [vol], tag=1)

    tag_volume_bdry(rect, [(tdim, vol)], model, factory)

    return size


def gUnitSquare(size):
    '''(0, 1)^2'''
    return g.Rectangle(np.array([0, 0]), np.array([1, 1]), size)


# @g4x.occ(1, 1)
# def gRectangleSurface(ll, ur, size, model, factory, view=False):
#     '''
#     Rectangle boundary marked by lower left and upper right corners. The returned
#     facet funtion is such that
#       4
#     1   2
#       3
#     If size is dict then we expact a mapping from tag to specs of Threshold
#     field in gmsh, i.e. SizeMax for size if thresholded > DistMax and 
#     analogously for SizeMin and DistMin
#     '''
#     ll, ur = np.array(ll), np.array(ur)

#     dx, dy = ur - ll
#     assert dx > 0 and dy > 0
#     # Point by point
#     # 4 3  
#     # 1 2
#     A = factory.addPoint(ll[0], ll[1], 0)
#     B = factory.addPoint(ur[0], ll[1], 0)
#     C = factory.addPoint(ur[0], ur[1], 0)
#     D = factory.addPoint(ll[0], ur[1], 0)

#     bottom = factory.addLine(A, B)
#     right = factory.addLine(B, C)
#     top = factory.addLine(C, D)
#     left = factory.addLine(D, A)
    
#     loop = factory.addCurveLoop([bottom, right, top, left])
#     rectangle = factory.addPlaneSurface([loop])

#     factory.synchronize()
    
#     bdry = (left, right, bottom, top)
#     for tag, curve in enumerate(bdry, 1):
#         model.addPhysicalGroup(1, [curve], tag)

#     return size

# # ---

# @g4x.occ(3, 2)
# def gBox(ll, ur, size, model, factory, view=False):
#     '''
#     Box marked by lower left and upper right corners. The returned
#     facet funtion is such that
#       4
#     1   2   5 z---> 6
#       3
#     If size is dict then we expact a mapping from tag to specs of Threshold
#     field in gmsh, i.e. SizeMax for size if thresholded > DistMax and 
#     analogously for SizeMin and DistMin
#     '''
#     ll, ur = np.array(ll), np.array(ur)

#     dx, dy, dz = ur - ll
#     assert dx > 0 and dy > 0 and dz > 0
    
#     box = factory.addBox(x=ll[0], y=ll[1], z=ll[2], dx=dx, dy=dy, dz=dz)

#     factory.synchronize()

#     model.addPhysicalGroup(3, [box], tag=1)

#     bdries = [(1, 0, ll), (2, 0, ur), (3, 1, ll), (4, 1, ur), (5, 2, ll), (6, 2, ur)]
#     # For marking look at centers of mass of the surfaces
#     while bdries:
#         tag, axis, point = bdries.pop()
#         surfs = iter(model.getEntities(2))
#         found = False
#         while not found:
#             surface = next(surfs)
#             com = factory.getCenterOfMass(*surface)
#             found = abs(com[axis] - point[axis]) < 1E-10
#         assert found
        
#         dim, index = surface
#         model.addPhysicalGroup(dim, [index], tag)

#     return size


# @g4x.occ(2, 2)
# def gBoxSurface(ll, ur, size, model, factory, view=False):
#     '''
#     Box boundary marked by lower left and upper right corners. The returned
#     facet funtion is such that
#       4
#     1   2   5 z---> 6
#       3
#     If size is dict then we expact a mapping from tag to specs of Threshold
#     field in gmsh, i.e. SizeMax for size if thresholded > DistMax and 
#     analogously for SizeMin and DistMin
#     '''
#     ll, ur = np.array(ll), np.array(ur)

#     dx, dy, dz = ur - ll
#     assert dx > 0 and dy > 0 and dz > 0

#     box = factory.addBox(x=ll[0], y=ll[1], z=ll[2], dx=dx, dy=dy, dz=dz)

#     factory.synchronize()

#     bdries = [(1, 0, ll), (2, 0, ur), (3, 1, ll), (4, 1, ur), (5, 2, ll), (6, 2, ur)]
#     # For marking look at centers of mass of the surfaces
#     while bdries:
#         tag, axis, point = bdries.pop()
#         surfs = iter(model.getEntities(2))
#         found = False
#         while not found:
#             surface = next(surfs)
#             com = factory.getCenterOfMass(*surface)
#             found = abs(com[axis] - point[axis]) < 1E-10
#         assert found
        
#         dim, index = surface
#         model.addPhysicalGroup(dim, [index], tag)

#     return size
