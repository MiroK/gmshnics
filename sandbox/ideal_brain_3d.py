import petsc4py, sys
petsc4py.init(sys.argv)

from gmshnics import msh_gmsh_model, mesh_from_gmsh
import itertools, collections
import meshio
import gmsh, os

from slepc4py import SLEPc
from petsc4py import PETSc
import dolfin as df
import numpy as np


Ellipsoid = collections.namedtuple('Ellipsoid', ('x0', 'y0', 'z0', 'a', 'b', 'c'))


def deform_ellipsoid(f, ellipsoid, scale=1., smooth=0, out='deformed_surface.stl', debug=False):
    '''
    Starting from an ellipsoid centered at (x0, y0, z0) with axis (a, b, c) 
    produce a surface stl file where each point of the ellipsoid surface is moved 
    in normal direction by f(theta, phi). 
    
    NOTE: f should take in the array npts x 2 where the columns are the angles
    '''
    # Create first the surface mesh; as a deformed sphere
    assert os.path.splitext(out)[1] == '.stl'
    
    gmsh.initialize(['', '-clscale', str(scale)])
    model = gmsh.model
    factory = model.occ

    x0, y0, z0, a, b, c = ellipsoid

    factory.addSphere(x0, y0, z0, 1)
    factory.synchronize()

    vol,  = model.getEntities(3)
    factory.dilate([vol], x0, y0, z0, a, b, c)
    factory.synchronize()

    surfs  = model.getEntities(2)
    model.addPhysicalGroup(2, [s[1] for s in surfs], 1)
    factory.synchronize()

    # NOTE: ideally we would get the access directly to nodes
    # in gmsh and manipulate those. However, we only can get 
    # copy and so to mode the nodes we will work with fenics mesh
    # and use meshio to get this into an STL file        
    nodes, topologies = msh_gmsh_model(model, 2)
    # Rely on gmsh to get the parametrization
    p = model.getParametrization(2, surfs[0][1], nodes.flatten())
    thetas, phis = p.reshape((-1, 2)).T

    mesh, _ = mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()
    
    # We should have x_i = x0_i + a_i*... For applying the
    # deformation we need normal and angles
    nodes = mesh.coordinates()
    x, y, z = nodes.T
    dx, dy, dz = (x-x0)/a, (y-y0)/b, (z - z0)/c
    
    n = 2*np.c_[dx, dy, dz]
    n /= np.linalg.norm(n, 2, axis=1).reshape((-1, 1))

    # Keep track of "mesh quality"
    triangles = mesh.cells()
    tri_area = lambda tri: 0.5*np.linalg.norm(np.cross(tri[1]-tri[0], tri[2]-tri[0]))

    h0 = min(map(tri_area, nodes[triangles]))
    
    # Shift the coordinates
    shift = f(np.c_[thetas, phis]).reshape((-1, 1))
    s0, s1 = np.min(shift.ravel()), np.max(shift.ravel())

    # Try to fix the pole singularity by smoothing
    if smooth > 0:
        # Try to get smoother shift
        cell = mesh.ufl_cell()
        Velm = df.FiniteElement('Lagrange', cell, 1)
        Qelm = df.FiniteElement('Real', cell, 0)
        Welm = df.MixedElement([Velm, Qelm])

        W = df.FunctionSpace(mesh, Welm)
        u, p = df.TrialFunctions(W)
        v, q = df.TestFunctions(W)

        # The data we want to smooth
        V = W.sub(0).collapse()
        order = df.vertex_to_dof_map(V)
        f = df.Function(V)
        arr = f.vector().get_local()
        arr[order] = shift.flatten()
        f.vector().set_local(arr)        

        kappa = df.Constant(smooth)
        # Deal with costant nullspace
        a = (df.inner(kappa*df.grad(u), df.grad(v))*df.dx + df.inner(p, v)*df.dx
             + df.inner(q, u)*df.dx)
        L = df.inner(f, v)*df.dx
        A, b = map(df.assemble, (a, L))

        # Preconditioner
        a_prec = (df.inner(kappa*df.grad(u), df.grad(v))*df.dx + df.inner(kappa*u, v)*df.dx
                  + df.inner(p, q)*df.dx)
        B = df.assemble(a_prec)

        solver = df.KrylovSolver('minres', 'hypre_amg')
        solver.set_operators(A, B)
        solver.parameters['relative_tolerance'] = 1E-20
        solver.parameters['absolute_tolerance'] = 1E-12
        solver.parameters['monitor_convergence'] = True

        wh = df.Function(W)
        solver.solve(wh.vector(), b)

        uh, _ = wh.split(deepcopy=True)
        shift_ = uh.vector().get_local()[order]
        # Finally we want the smooth shift to be in the same range
        # as the orignal one
        t = (shift_ - np.min(shift_))/(np.max(shift_) - np.min(shift_))
        shift = t*s0 + (1-t)*s1
        shift = shift.reshape((-1, 1))

    # Fro debugging it might be usefull to look at coordinates
    # and the resulting shift
    foos = []

    transforms = itertools.repeat(lambda x: x)
    
    if debug:
        V = df.FunctionSpace(df.Mesh(mesh), 'CG', 1)
        order = df.vertex_to_dof_map(V)

        for transform, vals in zip(transforms, (thetas, phis, shift)):
            f = df.Function(V)
            arr = f.vector().get_local()
            arr[order] = transform(vals.flatten())
            f.vector().set_local(arr)
            
            foos.append(f)

        V = df.VectorFunctionSpace(df.Mesh(mesh), 'CG', 1)
        order = df.vertex_to_dof_map(V)

        f = df.Function(V)
        arr = f.vector().get_local()
        arr[order] = transform((n*shift).flatten())
        f.vector().set_local(arr)

        foos.append(f)        
    
    nodes[:] += n*shift

    # Check that after deformation none of the elements vanished
    h = min(map(tri_area, nodes[triangles]))

    print(f'Min mesh size changes from {h0} to {h}')
    
    # Now we go to STL with meshio
    meshio.Mesh(mesh.coordinates(), [('triangle', mesh.cells())]).write(out)

    return out, foos


def mesh_stl_bounded_surface(path, scale=1, view=False):
    '''Collection of points encloses a surface'''
    gmsh.initialize(['', '-clscale', str(scale)])
    model = gmsh.model
    factory = model.geo

    gmsh.merge(path)
    factory.synchronize()

    if view:
        gmsh.fltk.initialize()
        gmsh.fltk.run()

    surf,  = model.getEntities(2)
    _, tag = surf
    l = factory.addSurfaceLoop([tag])
    v = factory.addVolume([l])

    factory.synchronize()
    
    model.addPhysicalGroup(3, [v], 1)

    nodes, topologies = msh_gmsh_model(model, 3)

    mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()

    return mesh
        
# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import dolfin as df
    import os

    f = lambda angles, k=4, l=5: 0.1*np.cos(2*k*angles[:, 0])*np.cos(2*l*angles[:, 1])

    ellipsoid = Ellipsoid(0, 0, 0, 1, 1, 1)
    # NOTE: might need to adjust scale which is the diffusion value
    # in the smoothing process
    stl, funcs = deform_ellipsoid(f, ellipsoid, smooth=1E-2, scale=0.1, debug=False)

    mesh = mesh_stl_bounded_surface(stl, view=False)
