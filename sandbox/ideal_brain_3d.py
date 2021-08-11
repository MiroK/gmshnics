import petsc4py, sys
petsc4py.init(sys.argv)

from gmshnics import msh_gmsh_model, mesh_from_gmsh
import meshio
import gmsh, os

from slepc4py import SLEPc
from petsc4py import PETSc
import dolfin as df
import numpy as np


def deform_ellipsoid(f, x0=0, y0=0, z0=0, a=1, b=1, c=1, scale=1., out='deformed_surface.stl'):
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
    n /= np.sqrt(dx**2 + dy**2 + dz**2).reshape((-1, 1))

    # Keep track of "mesh quality"
    triangles = mesh.cells()
    tri_area = lambda tri: 0.5*np.linalg.norm(np.cross(tri[1]-tri[0], tri[2]-tri[0]))

    h0 = min(map(tri_area, nodes[triangles]))
    
    # Shift the coordinates
    shift = f(np.c_[thetas, phis]).reshape((-1, 1))

    # Save prior to deformation
    V = df.FunctionSpace(df.Mesh(mesh), 'CG', 1)
    shift_f = df.Function(V)
    arr = shift_f.vector().get_local()
    arr[df.vertex_to_dof_map(V)] = shift.flatten()
    shift_f.vector().set_local(arr)
    
    nodes[:] += n*shift

    # Check that after deformation none of the elements vanished
    h = min(map(tri_area, nodes[triangles]))

    print(f'Min mesh size changes from {h0} to {h}')
    
    # Now we go to STL with meshio
    meshio.Mesh(mesh.coordinates(), [('triangle', mesh.cells())]).write(out)

    return out, shift_f

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

    f = lambda angles, k=3, l=2: 0.1*np.cos(2*k*angles[:, 0])#*np.cos(l*angles[:, 1])

    stl, shift_f = deform_ellipsoid(f, a=1, b=1, c=1, scale=0.1)

    df.File('foo.pvd') << shift_f
    
    mesh = mesh_stl_bounded_surface(stl, view=True)


