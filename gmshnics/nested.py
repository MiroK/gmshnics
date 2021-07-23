from gmshnics.tagging import tag_matched_entities
from gmshnics.utils import first, second
from gmshnics.interopt import occ
from gmshnics.shapes import Shape

import numpy as np
import itertools
import collections

# --------

# Here we infer dimension from the model dimension
@occ(-1, -1)
def _shape_bounding_shapes(outer, inner, size, model, factory, strict=True, view=False, tol=1E-10):
    '''A gmsh model of geometry where outer shape bounds inner shapes'''
    if isinstance(inner, Shape):
        inner = (inner, )
    assert isinstance(outer, Shape) and all(isinstance(i, Shape) for i in inner)
    # The outer bounds all
    assert not strict or all(i in outer for i in inner)
    # The inner guys are not nested ...
    assert not strict or not any(this in that or that in this for this, that in itertools.combinations(inner, 2))
    # ... and don't collide
    assert not strict or not any(this.intersects(that) for this, that in itertools.combinations(inner, 2))    

    # Create model
    outer_idx = outer.add(model, factory)
    inner_idx = [i.add(model, factory) for i in inner]
    factory.synchronize()
    
    # We better be consistent when it commes to dimensions
    tdim, = set(map(first, inner_idx))
    assert tdim == first(outer_idx)

    tag = factory.fragment([outer_idx], inner_idx)
    factory.synchronize()

    # In the order that the inner volume appeared let us tag them with
    # volumetric tags
    inner_shapes = list(reversed(inner))
    # Then each inner shape will also tag its boundaries and this look
    # up we want to keep
    surface_tags = collections.OrderedDict()
    volumes = model.getEntities(tdim)
    # By convention outer is 1 so for volumes we count from 2
    vtag = itertools.count(2)
    while inner_shapes:
        inner_shape = inner_shapes.pop()
        # We have only one match [(entity, physical tag)]
        [vol], volumes = tag_matched_entities([inner_shape.com], volumes, model, factory, vtag, tol)
        # So to the physical tag we temporaty assign the shape which we  will
        # later use to tag the boundary of this volume
        surface_tags[second(vol)] = [inner_shape]
    # By convention the outer volume receives tag 1; it is the left over volue
    vol_,  = volumes
    model.addPhysicalGroup(tdim, [second(vol_)], 1)
    factory.synchronize()
    surface_tags[second(vol_)] = [outer]
    
    stag = itertools.count(1)
    # Now in the order that inner volumes appeared we go for surfaces
    for vol in surface_tags:
        shape = surface_tags[vol].pop()
        # We want to match against the entities that are the volume boundary
        shape_bdry = list(model.getBoundary([(tdim, vol)]))
        # the shape surface coordinates
        surf_coms = shape.com_surfaces
        # We again get assoc list [(elementary_entities, physical_entities)]
        matched_tags, shape_bdry = tag_matched_entities(surf_coms, shape_bdry, model, factory, stag, tol)
        # Finally populate the look up in a right way
        surface_tags[vol].extend(map(second, matched_tags))

    return size#, surface_tags


def shape_bounding_shapes(outer, inner, size, view=False):
    '''Rectangles inside rectangle each speced by tuple that is ll and ur'''
    return _shape_bounding_shapes(outer=outer,
                                  inner=inner,
                                  size=size,
                                  view=view)


def rectangle_bounding_rectangles(outer, inner, size, view=False):
    '''Rectangles inside rectangle each speced by tuple that is ll and ur'''
    return shape_bounding_shapes(outer=Rectangle(*outer),
                                 inner=[Rectangle(*i) for i in inner],
                                 size=size,
                                 view=view)
            
# --------------------------------------------------------------------

if __name__ == '__main__':
    from gmshnics.shapes import Rectangle, Circle

    outer = ((0, 0), (1, 1))
    inner = (((0.25, 0.25), (0.5, 0.5)),
             ((0.55, 0.55), (0.75, 0.75)))

    inner = (((0.25, 0.25), (0.75, 0.75)), )
    # rectangle_bounding_rectangles(outer, inner, size=0.1, view=True)

    shape_bounding_shapes(outer=Rectangle((0, 0), (1, 1)),
                          inner=[Circle((0.5, 0.5), 0.2)],
                          size=0.2, view=True)
