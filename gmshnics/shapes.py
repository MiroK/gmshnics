import gmshnics.contouring as contour
import gmshnics.fractals as fractal
import numpy as np


class Shape:
    '''Gmsh shape for nested model'''
    tol = 1E-10
    
    def __init__(self, as_one_surface=False):
        self.as_one_surface = as_one_surface
        self._com = None
        self._com_surfaces = None
        self._vertices = None
    
    def is_inside(self, point):
        '''Check if point is inside shape'''
        pass

    def add(self, model, factory):
        '''Add self to gmsh model'''
        pass

    def __contains__(self, other):
        '''Is the shape contained inside other?'''
        return all(self.is_inside(p) for p in other.vertices)

    def intersects(self, other):
        '''Check collision with other shape'''
        return any(self.is_inside(p) for p in other.vertices)

    @property
    def com(self):
        '''Shape's center of mass'''
        return self._com

    @property
    def com_surfaces(self):
        '''Center of mass for each surface of the shape'''
        return self._com_surfaces if not self.as_one_surface else None

    @property
    def vertices(self):
        '''Vertices defining the shape'''
        return self._vertices


class Rectangle(Shape):
    '''Axis aligned rectangle is flat in z=0 plane'''
    def __init__(self, ll, ur, as_one_surface=False):
        super().__init__(as_one_surface)
        
        ll = np.fromiter(ll, dtype=float)
        ur = np.fromiter(ur, dtype=float)
        
        dx, dy = ur - ll
        assert dx > 0 and dy > 0
        # For collisions
        self.ll, self.ur = ll, ur

        vertices = np.array([[ll[0], ll[1]],
                             [ur[0], ll[1]],
                             [ur[0], ur[1]],
                             [ll[0], ur[1]]])

        self._com = np.mean(vertices, axis=0)

        edges = [(3, 0), (1, 2), (0, 1), (2, 3)]        
        self._com_surfaces = np.array([0.5*(vertices[i] + vertices[j]) for i, j in edges])

        self._vertices = vertices

    def is_inside(self, p):
        tol = Shape.tol
        return all((self.ll[0] + tol < p[0] < self.ur[0] - tol,
                    self.ll[1] + tol < p[1] < self.ur[1] - tol,
                    len(p) == 2 or abs(p[2]) < tol))

    def add(self, model, factory):
        ll, ur = self.ll, self.ur
        dx, dy = ur - ll
        return (2, factory.addRectangle(x=ll[0], y=ll[1], z=0, dx=dx, dy=dy))


class Box(Shape):
    '''Box'''
    def __init__(self, ll, ur, as_one_surface=False):
        super().__init__(as_one_surface)
        
        ll = np.fromiter(ll, dtype=float)
        ur = np.fromiter(ur, dtype=float)
        
        dx, dy, dz = ur - ll
        assert dx > 0 and dy > 0 and dz > 0
        # For collisions
        self.ll, self.ur = ll, ur

        vertices = np.array([[ll[0], ll[1], ll[2]],
                             [ur[0], ll[1], ll[2]],
                             [ur[0], ur[1], ll[2]],
                             [ll[0], ur[1], ll[2]],
                             [ll[0], ll[1], ur[2]],
                             [ur[0], ll[1], ur[2]],
                             [ur[0], ur[1], ur[2]],
                             [ll[0], ur[1], ur[2]]])

        self._com = np.mean(vertices, axis=0)

        facets = [[0, 3, 4, 7], [1, 2, 5, 6],
                  [0, 1, 4, 5], [2, 3, 7, 6],
                  [0, 1, 2, 3], [4, 5, 6, 7]]
        self._com_surfaces = np.array([np.mean(vertices[facet], axis=0) for facet in facets])

        self._vertices = vertices

    def is_inside(self, p):
        tol = Shape.tol
        return all((self.ll[i] + tol < p[i] < self.ur[i] - tol) for i in range(3))

    def add(self, model, factory):
        ll, ur = self.ll, self.ur
        dx, dy, dz = ur - ll
        return (3, factory.addBox(x=ll[0], y=ll[1], z=ll[2], dx=dx, dy=dy, dz=dz))
    
    
class Circle(Shape):
    '''Circle with center and radius'''
    def __init__(self, c, r, as_one_surface=False):
        super().__init__(as_one_surface)
        
        assert r > 0
        c = np.fromiter(c, dtype=float)
        assert len(c) == 2

        self.c, self.r = c, r

        thetas = 2*np.pi*np.linspace(0, 40)        
        self._vertices = c + r*np.c_[np.sin(thetas), np.cos(thetas)]
        
        self._com = c
        
        self._com_surfaces = np.array([self.c])

    def is_inside(self, p):
        return np.linalg.norm(p-self.c) < self.r-Shape.tol
               
    def add(self, model, factory):
        circle = factory.addCircle(x=self.c[0], y=self.c[1], z=0, r=self.r)
        loop = factory.addCurveLoop([circle])
        circle = factory.addPlaneSurface([loop])

        return (2, circle)


class Polygon(Shape):
    '''Closed one. Specified by it's LINKED vertices (no duplicates)'''
    tol = 1E-1
    def __init__(self, vertices, nrefs=4, as_one_surface=False):
        super().__init__(as_one_surface)
        nvtx, gdim = vertices.shape
        assert nvtx > 2 and gdim == 2

        self._vertices = vertices
        # For further computations it is convenien to have duplicate
        # the last vertex
        vertices = np.row_stack([vertices, vertices[0]])
        # Edges are just subsequent
        self._com_surfaces = 0.5*(vertices[:-1] + vertices[1:])
        # See https://stackoverflow.com/a/5271722
        x = np.c_[vertices[:-1, 0], vertices[1:, 0]]
        y = np.c_[vertices[:-1, 1], vertices[1:, 1]]
        x_cross_y = np.cross(x, y)
        area = 0.5*np.sum(x_cross_y)
        self._com = 2*(1/6/area)*np.array([np.sum(self._com_surfaces[:, 0]*x_cross_y),
                                           np.sum(self._com_surfaces[:, 1]*x_cross_y)])

        # Coarse outside filter is to kick out those outside of bounding box
        self.ll = np.min(vertices, axis=0)
        self.ur = np.max(vertices, axis=0)
        # For mode finer `is_inside` we will use wind number method; here we
        # precompute quadrature points for the integration
        self.windgen = contour.wind_number_gen(vertices, nrefs) 

    def is_inside(self, p):
        if p[0] > self.ur[0]+Shape.tol or p[0] < self.ll[0]-Shape.tol:
            return False

        if p[1] > self.ur[1]+Shape.tol or p[1] < self.ll[1]-Shape.tol:
            return False
        print('   ', p)
        # Compute the wind number and decide
        next(self.windgen)
        wn = self.windgen.send(p)
        
        print(abs(wn), self.tol, abs(wn) > self.tol)
        return abs(wn) > self.tol

    def add(self, model, factory):
        # Volume bounded be the contour
        pts = self.vertices
        
        l = len(pts)
        pts = [factory.addPoint(*p, z=0) for p in pts]        
        lines = [factory.addLine(pts[i], pts[(i+1)%l]) for i in range(len(pts))]
    
        loop = factory.addCurveLoop(lines)
        ngon = factory.addPlaneSurface([loop])        

        return (2, ngon)


class KochSnowflake(Polygon):
    tol = 1E-5
    def __init__(self, nsteps, initial=None, as_one_surface=False):
        vertices = fractal.koch_snowflake(nsteps, initial=initial)
        vertices = np.array(vertices)
        super().__init__(vertices, as_one_surface=as_one_surface)

# -------------------------------------------

if __name__ == '__main__':
    import gmsh

    gmsh.initialize()

    model = gmsh.model
    factory = model.occ

    f = KochSnowflake(2, as_one_surface=True)
    print(f.com)
    idx = f.add(model, factory)

    factory.synchronize()

    print(factory.getCenterOfMass(*idx))
    
    gmsh.fltk.initialize()
    gmsh.fltk.run()

    gmsh.finalize()
