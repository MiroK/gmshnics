# The idea of this construction is that model and factory are provided
# by the decorator. The decorated function makes a call to define the 
# geometry and based on fdim, tdim and size meshes the model. Not the
# most elegant solution but I am learning/trying something new ...
from gmshnics.shapes import Circle, Rectangle, Box, KochSnowflake
from gmshnics.tagging import tag_volume_bdry
from gmshnics.interopt import occ
import numpy as np


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


@occ(1, 1)
def gCircleSurface(center, radius, size, model, factory, view=False):
    '''
    Centered at center with radius. Returns mesh and CellFunction 
    with outer boundary set to 1
    '''
    circle = Circle(center, radius)
    tdim, vol = circle.add(model, factory)
    
    factory.synchronize()

    tag_volume_bdry(circle, [(tdim, vol)], model, factory)

    return size


def gUnitCircle(size, view=False):
    '''(0, 0) with 1'''
    return gCircle(np.array([0, 0]), 1, size, view=view)


def gUnitCircleSurface(size, view=False):
    '''(0, 0) with 1'''
    return gCircleSurface(np.array([0, 0]), 1, size, view=view)

# --

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


@occ(1, 1)
def gRectangleSurface(ll, ur, size, model, factory, view=False):
    '''
    Rectangle boundary marked by lower left and upper right corners. The returned
    cell funtion is such that
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

    tag_volume_bdry(rect, [(tdim, vol)], model, factory)

    return size


def gUnitSquare(size, view=False):
    '''(0, 1)^2'''
    return gRectangle(np.array([0, 0]), np.array([1, 1]), size, view=view)


def gUnitSquareSurface(size, view=False):
    '''partial (0, 1)^2'''
    return gRectangleSurface(np.array([0, 0]), np.array([1, 1]), size, view=view)

# ---

@occ(3, 2)
def gBox(ll, ur, size, model, factory, view=False):
    '''
    Box marked by lower left and upper right corners. The returned
    facet funtion is such that
      4
    1   2   5 z---> 6
      3
    If size is dict then we expact a mapping from tag to specs of Threshold
    field in gmsh, i.e. SizeMax for size if thresholded > DistMax and 
    analogously for SizeMin and DistMin
    '''
    box = Box(ll, ur)
    tdim, vol = box.add(model, factory)
    
    factory.synchronize()

    model.addPhysicalGroup(tdim, [vol], tag=1)

    tag_volume_bdry(box, [(tdim, vol)], model, factory)

    return size


@occ(2, 2)
def gBoxSurface(ll, ur, size, model, factory, view=False):
    '''
    Box boundary marked by lower left and upper right corners. The returned
    cell funtion is such that
      4
    1   2   5 z---> 6
      3
    If size is dict then we expact a mapping from tag to specs of Threshold
    field in gmsh, i.e. SizeMax for size if thresholded > DistMax and 
    analogously for SizeMin and DistMin
    '''
    box = Box(ll, ur)
    tdim, vol = box.add(model, factory)
    
    factory.synchronize()

    tag_volume_bdry(box, [(tdim, vol)], model, factory)

    return size


def gUnitCube(size, view=False):
    return gBox(np.array([0, 0, 0]), np.array([1, 1, 1]), size=size, view=view)


def gUnitCubeSurface(size, view=False):
    return gBoxSurface(np.array([0, 0, 0]), np.array([1, 1, 1]), size=size, view=view)


@occ(2, 1)
def gKochSnowFlake(nsteps, initial, size, model, factory, view=False, as_one_surface=True):
    '''
    Centered at center with radius. Returns mesh and FacetFunction 
    with outer boundary set to 1
    '''
    flake = KochSnowflake(nsteps, initial)
    tdim, vol = flake.add(model, factory)
    
    factory.synchronize()

    model.addPhysicalGroup(tdim, [vol], tag=1)

    tag_volume_bdry(flake, [(tdim, vol)], model, factory)

    return size


@occ(2, 1)
def gDisk(center, inradius, outradius, size, model, factory, view=False):
    '''
    Centered at center with (inradius, outradius). Returns mesh and FacetFunction 
    with inner boundary set to 1, outer to 2.
    '''
    assert 0 < inradius < outradius

    icircle = Circle(center, inradius)
    tdim, ivol = icircle.add(model, factory)

    ocircle = Circle(center, outradius)
    tdim, ovol = ocircle.add(model, factory)
    
    factory.synchronize()

    disk, _ = factory.cut([(tdim, ovol)], [(tdim, ivol)])

    model.addPhysicalGroup(2, [disk[0][1]], 1)
    model.addPhysicalGroup(1, [1], 1)
    model.addPhysicalGroup(1, [2], 2)

    return size

# ------

if __name__ == "__main__":
    gKochSnowFlake(nsteps=4, initial=None, size=0.2, view=True)
