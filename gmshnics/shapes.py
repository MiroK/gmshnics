import numpy as np


class Shape:
    '''Gmsh shape for nested model'''
    tol = 1E-10
    
    def __init__(self, break_up_surfaces=False):
        self.break_up_surfaces = False
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
        return self._com_surfaces if not self.break_up_surfaces else None

    @property
    def vertices(self):
        '''Vertices defining the shape'''
        return self._vertices


class Rectangle(Shape):
    '''Axis aligned rectangle is flat in z=0 plane'''
    def __init__(self, ll, ur, break_up_surfaces=False):
        super().__init__(break_up_surfaces)
        
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

    
class Circle(Shape):
    '''Circle with center and radius'''
    def __init__(self, c, r, break_up_surfaces=False):
        super().__init__(break_up_surfaces)
        
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
