import gmshnics as g4x
import numpy as np
import gmsh


def gCircle(center, radius, size, view=False):
    '''
    Centered at center with radius. Returns mesh and FacetFunction 
    with outer boundary set to 1
    '''
    gmsh.initialize()
    
    model = gmsh.model
    factory = model.occ

    cx, cy = center
    circle = factory.addCircle(cx, cy, 0, radius)
    loop = factory.addCurveLoop([circle])
    circle = factory.addPlaneSurface([loop])

    factory.synchronize()

    model.addPhysicalGroup(2, [circle], tag=1)
    
    bdry = model.getBoundary([(2, circle)])

    for tag, curve in enumerate(bdry, 1):
        model.addPhysicalGroup(1, [curve[1]], tag)

    factory.synchronize()    

    nodes, topologies = g4x.msh_gmsh_model(model,
                                           2,
                                           # Globally refine
                                           number_options={'Mesh.CharacteristicLengthFactor': size},
                                           view=view)

    mesh, entity_functions = g4x.mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()

    return mesh, entity_functions[1]


def gRectangle(ll, ur, size, view=False):
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
    ll, ur = np.array(ll), np.array(ur)

    dx, dy = ur - ll
    assert dx > 0 and dy > 0
    
    gmsh.initialize()
    
    model = gmsh.model
    factory = model.occ

    # Point by point
    # 4 3  
    # 1 2
    A = factory.addPoint(ll[0], ll[1], 0)
    B = factory.addPoint(ur[0], ll[1], 0)
    C = factory.addPoint(ur[0], ur[1], 0)
    D = factory.addPoint(ll[0], ur[1], 0)

    bottom = factory.addLine(A, B)
    right = factory.addLine(B, C)
    top = factory.addLine(C, D)
    left = factory.addLine(D, A)
    
    loop = factory.addCurveLoop([bottom, right, top, left])
    rectangle = factory.addPlaneSurface([loop])

    factory.synchronize()

    model.addPhysicalGroup(2, [rectangle], tag=1)
    
    bdry = (left, right, bottom, top)
    for tag, curve in enumerate(bdry, 1):
        model.addPhysicalGroup(1, [curve], tag)

    factory.synchronize()    

    if isinstance(size, dict):
        number_options = None
        # Set size field
        assert size.keys() <= set((1, 2, 3, 4))

        curves = {tag: (c, ) for tag, c in enumerate(bdry, 1)}
        set_facet_distance_field(size, curves, model, factory, 1)
    else:
        number_options = {'Mesh.CharacteristicLengthFactor': size}

    nodes, topologies = g4x.msh_gmsh_model(model,
                                           2,
                                           number_options=number_options,
                                           view=view)

    mesh, entity_functions = g4x.mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()

    return mesh, entity_functions[1]


def gRectangleSurface(ll, ur, size, view=False):
    '''
    Rectangle boundary marked by lower left and upper right corners. The returned
    facet funtion is such that
      4
    1   2
      3
    If size is dict then we expact a mapping from tag to specs of Threshold
    field in gmsh, i.e. SizeMax for size if thresholded > DistMax and 
    analogously for SizeMin and DistMin
    '''
    ll, ur = np.array(ll), np.array(ur)

    dx, dy = ur - ll
    assert dx > 0 and dy > 0
    
    gmsh.initialize()
    
    model = gmsh.model
    factory = model.occ

    # Point by point
    # 4 3  
    # 1 2
    A = factory.addPoint(ll[0], ll[1], 0)
    B = factory.addPoint(ur[0], ll[1], 0)
    C = factory.addPoint(ur[0], ur[1], 0)
    D = factory.addPoint(ll[0], ur[1], 0)

    bottom = factory.addLine(A, B)
    right = factory.addLine(B, C)
    top = factory.addLine(C, D)
    left = factory.addLine(D, A)
    
    loop = factory.addCurveLoop([bottom, right, top, left])
    rectangle = factory.addPlaneSurface([loop])

    factory.synchronize()
    
    bdry = (left, right, bottom, top)
    for tag, curve in enumerate(bdry, 1):
        model.addPhysicalGroup(1, [curve], tag)

    factory.synchronize()    

    if isinstance(size, dict):
        number_options = None
        # Set size field
        assert size.keys() <= set((1, 2, 3, 4))

        curves = {tag: (c, ) for tag, c in enumerate(bdry, 1)}
        set_facet_distance_field(size, curves, model, factory, 1)
    else:
        number_options = {'Mesh.CharacteristicLengthFactor': size}

    nodes, topologies = g4x.msh_gmsh_model(model,
                                           1,
                                           number_options=number_options,
                                           view=view)

    mesh, entity_functions = g4x.mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()

    return mesh, entity_functions[1]

# ---

def gBox(ll, ur, size, view=False):
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
    ll, ur = np.array(ll), np.array(ur)

    dx, dy, dz = ur - ll
    assert dx > 0 and dy > 0 and dz > 0
    
    gmsh.initialize()
    
    model = gmsh.model
    factory = model.occ

    box = factory.addBox(x=ll[0], y=ll[1], z=ll[2], dx=dx, dy=dy, dz=dz)

    factory.synchronize()

    model.addPhysicalGroup(3, [box], tag=1)

    bdries = [(1, 0, ll), (2, 0, ur), (3, 1, ll), (4, 1, ur), (5, 2, ll), (6, 2, ur)]
    # For marking look at centers of mass of the surfaces
    while bdries:
        tag, axis, point = bdries.pop()
        surfs = iter(model.getEntities(2))
        found = False
        while not found:
            surface = next(surfs)
            com = factory.getCenterOfMass(*surface)
            found = abs(com[axis] - point[axis]) < 1E-10
        assert found
        
        dim, index = surface
        model.addPhysicalGroup(dim, [index], tag)

    factory.synchronize()    

    if isinstance(size, dict):
        number_options = None
        # Set size field
        assert size.keys() <= set((1, 2, 3, 4, 5, 6))

        surfs = {tag: (c[1], ) for tag, c in enumerate(model.getEntities(2), 1)}
        set_facet_distance_field(size, surfs, model, factory, 2)
    else:
        number_options = {'Mesh.CharacteristicLengthFactor': size}

    nodes, topologies = g4x.msh_gmsh_model(model,
                                           3,
                                           number_options=number_options,
                                           view=view)

    mesh, entity_functions = g4x.mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()

    return mesh, entity_functions[2]


def gBoxSurface(ll, ur, size, view=False):
    '''
    Box boundary marked by lower left and upper right corners. The returned
    facet funtion is such that
      4
    1   2   5 z---> 6
      3
    If size is dict then we expact a mapping from tag to specs of Threshold
    field in gmsh, i.e. SizeMax for size if thresholded > DistMax and 
    analogously for SizeMin and DistMin
    '''
    ll, ur = np.array(ll), np.array(ur)

    dx, dy, dz = ur - ll
    assert dx > 0 and dy > 0 and dz > 0
    
    gmsh.initialize()
    
    model = gmsh.model
    factory = model.occ

    box = factory.addBox(x=ll[0], y=ll[1], z=ll[2], dx=dx, dy=dy, dz=dz)

    factory.synchronize()

    bdries = [(1, 0, ll), (2, 0, ur), (3, 1, ll), (4, 1, ur), (5, 2, ll), (6, 2, ur)]
    # For marking look at centers of mass of the surfaces
    while bdries:
        tag, axis, point = bdries.pop()
        surfs = iter(model.getEntities(2))
        found = False
        while not found:
            surface = next(surfs)
            com = factory.getCenterOfMass(*surface)
            found = abs(com[axis] - point[axis]) < 1E-10
        assert found
        
        dim, index = surface
        model.addPhysicalGroup(dim, [index], tag)

    factory.synchronize()    

    if isinstance(size, dict):
        number_options = None
        # Set size field
        assert size.keys() <= set((1, 2, 3, 4, 5, 6))

        surfs = {tag: (c[1], ) for tag, c in enumerate(model.getEntities(2), 1)}
        set_facet_distance_field(size, surfs, model, factory, 2)
    else:
        number_options = {'Mesh.CharacteristicLengthFactor': size}

    nodes, topologies = g4x.msh_gmsh_model(model,
                                           2,
                                           number_options=number_options,
                                           view=view)

    mesh, entity_functions = g4x.mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()

    return mesh, entity_functions[2]

# --------------------------------------------------------------------

def set_facet_distance_field(sizes, facets, model, factory, tdim):
    '''
    Set mesh size specifying for each physical facet group the gmsh
    Threshold field. Here facets maps phys tag to list of facet indices
    '''
    assert all(v.keys() == set(('SizeMax', 'DistMax', 'SizeMin', 'DistMin'))
               for v in sizes.values())

    field = model.mesh.field

    facets_list = {1: 'CurvesList', 2: 'SurfacesList'}[tdim]
        
    field_tag = 0
    thresholds = []
    for phys_tag, facet_sizes in sizes.items():
        field_tag += 1
        field.add('Distance', field_tag)
        field.setNumbers(field_tag, facets_list, facets[phys_tag])
        field.setNumber(field_tag, 'NumPointsPerCurve', 100)

        field_tag += 1
        field.add('Threshold', field_tag)
        field.setNumber(field_tag, 'InField', field_tag-1)
        # Set spec
        for prop in facet_sizes:
            field.setNumber(field_tag, prop, facet_sizes[prop])
        # Collect for setting final min
        thresholds.append(field_tag)

    min_field_tag = max(thresholds) + 1
    field.add('Min', min_field_tag)
    field.setNumbers(min_field_tag, 'FieldsList', thresholds)    
    field.setAsBackgroundMesh(min_field_tag)

    factory.synchronize()

    return min_field_tag
