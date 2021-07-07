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
        assert all(v.keys() == set(('SizeMax', 'DistMax', 'SizeMin', 'DistMin'))
                   for v in size.values())

        field = model.mesh.field
        
        field_tag = 0
        thresholds = []
        for phys_tag, curve_sizes in size.items():
            field_tag += 1
            field.add('Distance', field_tag)
            print(bdry, bdry[phys_tag-1], phys_tag)
            field.setNumbers(field_tag, 'CurvesList', [bdry[phys_tag-1]])
            field.setNumber(field_tag, 'NumPointsPerCurve', 100)

            field_tag += 1
            field.add('Threshold', field_tag)
            field.setNumber(field_tag, 'InField', field_tag-1)
            # Set spec
            for prop in curve_sizes:
                field.setNumber(field_tag, prop, curve_sizes[prop])
            # Collect for setting final min
            thresholds.append(field_tag)

        min_field_tag = max(thresholds) + 1
        field.add('Min', min_field_tag)
        field.setNumbers(min_field_tag, 'FieldsList', thresholds)    
        field.setAsBackgroundMesh(min_field_tag)

        factory.synchronize()
    else:
        number_options = {'Mesh.CharacteristicLengthFactor': size}

    nodes, topologies = g4x.msh_gmsh_model(model,
                                           2,
                                           number_options=number_options,
                                           view=view)

    mesh, entity_functions = g4x.mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()

    return mesh, entity_functions[1]
