from gmshnics import msh_gmsh_model, mesh_from_gmsh
import numpy as np
import gmsh


def perturb_ellipse(f, npts, x0=0, y0=0, a=1, b=1):
    '''
    Given f: (0, 2*pi) -> R we perturb ellipse centered at (x0, y0) with 
    axes a, b by walking distance f(theta) in direction of the unit normal 
    vector
    '''
    thetas = 2*np.pi*np.linspace(0, 1, npts)
    # The ellipse
    x = x0 + a*np.cos(thetas)
    y = y0 + b*np.sin(thetas)
    # Tangent
    dx = -a*np.sin(thetas)
    dy = b*np.cos(thetas)

    dl = np.sqrt(dx**2 + dy**2)
    # Unit normal
    u = dy/dl
    v = -dx/dl

    X = x + u*f(thetas)
    Y = y + v*f(thetas)

    return np.c_[X, Y], np.c_[x, y]


def mesh_contour(contour, scale=1, fit_spline=True, view=False):
    '''Collection of points encloses a surface'''
    npts, gdim = contour.shape
    assert npts > 3
    assert np.linalg.norm(contour[0] - contour[-1]) < 1E-13
    
    gmsh.initialize(['', '-clscale', str(scale)])
    model = gmsh.model
    factory = model.occ

    mapping = [factory.addPoint(*xi, z=0) for xi in contour[:-1]]
    mapping.append(mapping[0])
    if fit_spline:
        lines = [factory.addBSpline(mapping)]
    else:
        lines = [factory.addLine(*l) for l in zip(mapping[:-1], mapping[1:])]
    
    factory.synchronize()
    model.addPhysicalGroup(1, lines, 1)

    curve_loops = []
    curve_loops.append(factory.addCurveLoop(lines))
        
    surf = factory.addPlaneSurface(curve_loops)
    factory.synchronize()
    model.addPhysicalGroup(2, [surf], 1)
        
    factory.synchronize()

    if view:
        gmsh.fltk.initialize()
        gmsh.fltk.run()

    nodes, topologies = msh_gmsh_model(model, 2)

    mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()

    return mesh

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import dolfin as df
    
    f = lambda th: 0.1*np.cos(10*th)
    
    contour, ellipse = perturb_ellipse(f, npts=1000, x0=0, y0=0, a=1, b=1.5)

    mesh = mesh_contour(contour, scale=0.25, fit_spline=True, view=False)

    df.File('ibrain.pvd') << mesh
    
    plt.figure()
    plt.plot(ellipse[:, 0], ellipse[:, 1])    
    plt.plot(contour[:, 0], contour[:, 1])
    plt.axis('equal')
    plt.show()
