import petsc4py, sys
petsc4py.init(sys.argv)

from gmshnics import msh_gmsh_model, mesh_from_gmsh
from collections import defaultdict
import itertools
import gmsh

from slepc4py import SLEPc
from petsc4py import PETSc
import dolfin as df
import numpy as np



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


def force_per_area(V):
    '''
    What is |grad(u*).n|^2 if -Delta u* = 0 and u* = 1 on the boundary
    '''
    u, v = df.TrialFunction(V), df.TestFunction(V)

    a = df.inner(df.grad(u), df.grad(v))*df.dx
    L = df.inner(df.Constant(1), v)*df.dx
    bcs = df.DirichletBC(V, df.Constant(0), 'on_boundary')

    A, b = df.assemble_system(a, L, bcs)

    solver = df.KrylovSolver('cg', 'hypre_amg')
    solver.set_operators(A, A)
    solver.parameters['absolute_tolerance'] = 1E-12
    solver.parameters['relative_tolerance'] = 1E-20
    solver.parameters['monitor_convergence'] = True

    uh = df.Function(V)
    solver.solve(uh.vector(), b)

    mesh = V.mesh()
    n = df.FacetNormal(mesh)
    # Now what we are after
    area = df.assemble(df.Constant(1)*df.ds(domain=mesh))
    force = df.sqrt(abs(df.assemble(df.inner(df.dot(df.grad(uh), n), df.dot(df.grad(uh), n))*df.ds)))

    return force/area, df.as_backend_type(uh.vector()).vec()


def poincare_constant(V):
    '''
    Get Poincare constant as a largest eigenvalue of (u, v)*dx = l*(grad(u), grad(v))*dx
    '''
    u, v = df.TrialFunction(V), df.TestFunction(V)

    a = df.inner(df.grad(u), df.grad(v))*df.dx
    m = df.inner(u, v)*df.dx
    L = df.inner(df.Constant(0), v)*df.dx

    bcs = df.DirichletBC(V, df.Constant(0), 'on_boundary')

    z = df.Function(V).vector()
    # Want to build basis for space to avoid duting eigenvalue considerations
    # here it is does on boundary
    bdry_dofs = bcs.get_boundary_values()
    values = np.zeros(V.dim())

    basis = []
    for dof in bdry_dofs:
        values[dof] = 1.0
        
        z.set_local(values)
        basis.append(df.as_backend_type(z).vec().copy())
        # Reset
        values[dof] = 0.0

    A, _ = df.assemble_system(a, L, bcs)
    B, _ = df.assemble_system(m, L, bcs)

    A, B = (df.as_backend_type(x).mat() for x in (A, B))

    return largest_eigenvalue(B, A, basis)


def trace_constant(V):
    '''
    Get constant from trace inequality as a largest eigenvalue of 

      (u, v)*ds = l*(grad(u), grad(v))*dx

    '''
    u, v = df.TrialFunction(V), df.TestFunction(V)

    a = df.inner(df.grad(u), df.grad(v))*df.dx + df.inner(u, v)*df.dx
    m = df.inner(u, v)*df.ds

    bcs = df.DirichletBC(V, df.Constant(0), 'on_boundary')

    z = df.Function(V).vector()
    # Want to build basis for space to avoid duting eigenvalue considerations
    # here it is does on boundary
    bdry_dofs = bcs.get_boundary_values()
    values = np.zeros(V.dim())

    # basis = [df.as_backend_type(df.interpolate(df.Constant(1), V).vector()).vec()]

    A, B = map(df.assemble, (a, m))
    A, B = (df.as_backend_type(x).mat() for x in (A, B))

    return largest_eigenvalue(B, A)


def largest_eigenvalue(A, B, Z=None):
    '''Solve Ax = l B*x for the largest eigenvalue'''
    opts = PETSc.Options()
    opts.setValue('eps_rtol', 1E-8)
    opts.setValue('eps_max_it', 20000)
    opts.setValue('eps_nev', 1)
    opts.setValue('eps_monitor', None)
    opts.setValue('eps_type', 'krylovschur')
    opts.setValue('st_ksp_rtol', 1E-8)
    opts.setValue('st_ksp_monitor_true_residual', None)

    # Setup the eigensolver
    E = SLEPc.EPS().create()
    Z is not None and E.setDeflationSpace(Z)

    E.setOperators(A, B)

    E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)

    if A.size[0] > 2E6:
        ST = E.getST()
        ST.setType('sinvert')
        KSP = ST.getKSP()
        KSP.setType('cg')  # How to invert the B matrix
        PC = KSP.getPC()

        PC.setType('lu')
        PC.setFactorSolverPackage('mumps')

        KSP.setFromOptions()
    E.setFromOptions()
    
    E.solve()

    its = E.getIterationNumber()
    nconv = E.getConverged()
    
    eigw, i = max([(E.getEigenvalue(i), i) for i in range(nconv)],
                  key=lambda p: np.abs(p[0]))

    eigv = A.createVecLeft()
    E.getEigenvector(i, eigv)

    return eigw, eigv


def volume(omega):
    '''|Omega|'''
    return df.assemble(df.Constant(1)*df.dx(domain=omega))


def surface_area(omega):
    '''|partial Omega|'''
    return df.assemble(df.Constant(1)*df.ds(domain=omega))


def randomize(f, zero_bdry):
    '''A random function'''
    f.vector().set_local(np.random.randn(f.vector().local_size()))

    if zero_bdry:
        bc = df.DirichletBC(f.function_space(), df.Constant(np.zeros(f.ufl_shape)), 'on_boundary')
        bc.apply(f.vector())

    return f


def L2_norm(f):
    '''||f||_0'''
    return df.sqrt(abs(df.assemble(df.inner(f, f)*df.dx)))


def H10_norm(f):
    '''||f'||_0'''
    return df.sqrt(abs(df.assemble(df.inner(df.grad(f), df.grad(f))*df.dx)))


def constant_estimate(contour, max_nrefs, get_constant, tol=1E-2):
    '''Refine the contour mesh computing constants'''
    if not isinstance(get_constant, dict):
        which = {'foo': get_constant}
        return constant_estimate(contour, resolutions, get_constant)

    GREEN = '\033[1;37;32m%s\033[0m'        

    estimates, modes = defaultdict(list), {}
    status = dict(zip(get_constant, itertools.repeat(False)))
    not_converged = lambda k: not status[k]
    
    k = 0
    while any(filter(not_converged, status)) and k < max_nrefs:
        k += 1
        
        mesh = mesh_contour(contour, scale=1./2**k, fit_spline=True, view=False)
        V = df.FunctionSpace(mesh, 'CG', 1)

        hmin, Vdim = mesh.hmin(), V.dim()
        vol, surf = volume(mesh), surface_area(mesh)

        # Consider those that are not converged
        for name in filter(not_converged, get_constant):
            get_constant_f = get_constant[name]
            lmbda, mode_ = get_constant_f(V)
            lmbda = lmbda.real
        
            this_estimates = estimates[name]
            this_estimates.append((hmin, Vdim, lmbda, vol, surf))

            diff = np.nan
            if len(this_estimates) > 1:
                l0, l1 = this_estimates[-2], this_estimates[-1]
                # Relative difference
                diff = abs((l1[2]-l0[2])/l0[2])
                status[name] = diff < tol
            else:
                status[name] = False

            msg = f'{k} {name} {lmbda}[{diff}] {vol} {surf}'
            print(GREEN % msg)

            mode = df.Function(V)
            mode.vector()[:] = df.PETScVector(mode_)
            modes[name] = mode

    return estimates, modes
        
# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import dolfin as df
    import os

    root = './results'
    not os.path.exists(root) and os.mkdir(root)

    max_nrefs = 10

    get_constant = {'poincare': poincare_constant,
                    'trace': trace_constant,
                    'force': force_per_area} 

    for k in (0, 2, 4, 8, 16, 32):
        if k == 0:
            f = lambda th: np.zeros_like(th)
        else:
            f = lambda th, k=k: 0.1*np.cos(k*th)
            
        contour, ellipse = perturb_ellipse(f, npts=4000, x0=0, y0=0, a=1, b=1.5)

        results, _ = constant_estimate(contour, max_nrefs=max_nrefs, get_constant=get_constant)

        for name in results:
            np.savetxt(os.path.join(root, f'{name}_{k}.txt'), np.array(results[name]),
                       header='hmin Vdim lmbda vol surf')

