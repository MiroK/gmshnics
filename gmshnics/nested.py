from gmshnics.utils import first, second
import numpy as np
import itertools
import collections


class Shape:
    '''Gmsh shape for nested model'''
    tol = 1E-10
    
    def __contains__(self, other):
        '''Is the shape contained inside other?'''
        return all(self.is_inside(p) for p in other.vertices())

    def intersects(self, other):
        '''Check collision with other shape'''
        return any(self.is_inside(p) for p in other.vertices())

    def is_inside(self, point):
        '''Check if point is inside shape'''
        pass

    def add(self, model, factory):
        '''Add self to gmsh model'''
        pass

    def com(self):
        '''Shape's center of mass'''
        pass
    
    def com_surfaces(self):
        '''Center of mass for each surface of the shape'''
        pass

    def vertices(self):
        '''Vertices defining the shape'''
        pass


class Rectangle(Shape):
    '''Axis aligned rectangle is flat in z=0 plane'''
    def __init__(self, ll, ur):
        ll = np.fromiter(ll, dtype=float)
        ur = np.fromiter(ur, dtype=float)
        
        dx, dy = ur - ll
        assert dx > 0 and dy > 0

        self.ll, self.ur = ll, ur

    def vertices(self):
        ll, ur = self.ll, self.ur
        return np.array([[ll[0], ll[1]],
                         [ur[0], ll[1]],
                         [ur[0], ur[1]],
                         [ll[0], ur[1]]])

    def com(self):
        return np.mean(self.vertices(), axis=0)

    def com_surfaces(self):
        vs = self.vertices()
        # For i in axis then min/max
        edges = [(3, 0), (1, 2), (0, 1), (2, 3)]
        return np.array([0.5*(vs[i] + vs[j]) for i, j in edges])

    def is_inside(self, p):
        tol = Shape.tol
        return all((self.ll[0] + tol < p[0] < self.ur[0] - tol,
                    self.ll[1] + tol < p[1] < self.ur[1] - tol,
                    len(p) == 2 or abs(p[2]) < tol))

    def add(self, model, factory):
        ll, ur = self.ll, self.ur
        dx, dy = ur - ll
        return (2, factory.addRectangle(x=ll[0], y=ll[1], z=0, dx=dx, dy=dy))


def pair_up(entities, x, factory, tol):
    '''Find entity mathcing x in center of mass'''
    found = False
    while not found:
        for entity in entities:
            found = np.linalg.norm(factory.getCenterOfMass(*entity)[:len(x)] - x) < tol
            if found: break
    assert found
    
    return entity


def shape_bounding_shapes(outer, inner, view=True, tol=1E-10):
    '''A gmsh model of geometry where outer shape bounds inner shapes'''
    if isinstance(inner, Shape):
        inner = (inner, )
    assert isinstance(outer, Shape) and all(isinstance(i, Shape) for i in inner)
    # The outer bounds all
    assert all(i in outer for i in inner)
    # The inner guys are not nested ...
    assert not any(this in that or that in this for this, that in itertools.combinations(inner, 2))
    # ... and don't collide
    assert not any(this.intersects(that) for this, that in itertools.combinations(inner, 2))    

    # Let's define the geometry
    gmsh.initialize()

    model = gmsh.model
    factory = model.occ

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
    volumes = list(model.getEntities(tdim))
    # By convention outer is 1 so for volumes we count from 2
    vtag = itertools.count(2)
    while inner_shapes:
        inner_shape = inner_shapes.pop()
        com = inner_shape.com()
        # Find a match based on center of mass
        vol = pair_up(volumes, com, factory, tol)
        volumes.remove(vol)
        # Set vol tag
        model.addPhysicalGroup(tdim, [second(vol)], next(vtag))
        factory.synchronize()
        # We will fill them proper later when looping
        surface_tags[second(vol)] = [inner_shape]
    # By congention the outer volume receives tag 1
    vol_,  = volumes
    model.addPhysicalGroup(tdim, [second(vol_)], 1)
    factory.synchronize()
    surface_tags[second(vol_)] = [outer]
    
    stag = itertools.count(1)
    for vol in surface_tags:
        shape = surface_tags[vol].pop()
        
        shape_bdry = list(model.getBoundary([(tdim, vol)]))
        
        fdim, = set(map(first, shape_bdry))
        # Now the shape is going to claim it surfaces. 
        surface_coms = shape.com_surfaces()
        # If they are not made specific by shape then entire boundary gets one tag
        if surface_coms is None or not len(surface_coms):
            model.addPhysicalGroup(fdim, list(map(second, shape_bdry)), next(stag))
        else:
            scoms = list(reversed(surface_coms))
            while scoms:
                scom = scoms.pop()
    
                surf = pair_up(shape_bdry, scom, factory, tol)
                shape_bdry.remove(surf)
                # Set surface tag
                stag_ = next(stag)
                model.addPhysicalGroup(fdim, [second(surf)], stag_)
                factory.synchronize()
                
                surface_tags[vol].append(stag_)
    
    if view:
        gmsh.fltk.initialize()
        gmsh.fltk.run()

    gmsh.finalize()
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    import gmsh

    outer = Rectangle((0, 0), (1, 1))

    inner = [Rectangle((0.25, 0.25), (0.5, 0.5)),
             Rectangle((0.55, 0.55), (0.75, 0.75))]

    shape_bounding_shapes(outer, inner, view=True)    
