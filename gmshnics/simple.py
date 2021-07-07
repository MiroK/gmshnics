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
    A = factory.addPoint(ll[0], ll[0], 0)
    B = factory.addPoint(ur[0], ll[0], 0)
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

    nodes, topologies = g4x.msh_gmsh_model(model,
                                           2,
                                           # Globally refine
                                           number_options={'Mesh.CharacteristicLengthFactor': size},
                                           view=view)

    mesh, entity_functions = g4x.mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()

    return mesh, entity_functions[1]

# --------------------------------------------------------------------

if __name__ == '__main__':

    gCircle(center=(0, 0), radius=1., size=0.1)
    gRectangle((0, 0), (1, 2), size=0.1, view=True)

    
