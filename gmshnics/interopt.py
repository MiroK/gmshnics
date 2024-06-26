from gmshnics.make_mesh_cpp import build_mesh
from gmshnics.utils import second
from gmshnics.meshsize import set_facet_distance_field

from collections import defaultdict
from functools import reduce, wraps

import dolfin as df
import numpy as np
import operator
import gmsh
try: 
    import ufl
except ImportError:
    import ufl_legacy as ufl


def occ(tdim, fdim):
    '''
    Mesh the tdim entities of the model use optionally as size field 
    the distance to fdim-entities
    '''
    def mesh_it(f):
        @wraps(f)
        def wrapper(*args, **kwds):
            gmsh.initialize()
            model = gmsh.model
            factory = model.occ

            _tdim, _fdim = tdim, fdim
            # Create model expecting size info, model and factory are
            # looked up in this scope
            kwds.update({'model': model, 'factory': factory})
            retval = f(*args, **kwds)
            # Typically models return just size but in embedded models we get lookup
            # for facets of shapes involved. This is useful an we want to pass it outside
            return_lookup = isinstance(retval, tuple)
            if return_lookup:
                size, lookup = retval
            else:
                size = retval
            
            if _tdim == -1 and _fdim == -1:
                _tdim = model.getDimension()
                # Most we have in mind situtations where the size field is set on
                # distances from facets
                _fdim = _tdim - 1
            # Mesh using size info
            factory.synchronize()

            number_options = kwds.get('number_options', None)
            if number_options is None:
                number_options = {}
                
            string_options = kwds.get('string_options', None)
            if string_options is None:
                string_options = {}

            if isinstance(size, dict):
                # We want to remove size stuff from number options and
                # set size field
                if 'Mesh.CharacteristicLengthFactor' in number_options:
                    del number_options['Mesh.CharacteristicLengthFactor']
                # The size dict refers to entities by their physical group
                # so let's get those allowed
                facets = defaultdict(list)

                [facets[ptag].append(second(entity))
                 for entity in model.getEntities(_fdim)
                 for ptag in model.getPhysicalGroupsForEntity(_fdim, second(entity))]
                
                assert size.keys() <= facets.keys()
                print(size, facets)
                fid = set_facet_distance_field(size, facets, model, factory, _fdim)
                print(f'MeshField used as background is {fid}')
            # Otherwise we just a number for char size
            else:
                number_options['Mesh.CharacteristicLengthFactor'] = size

            nodes, topologies = msh_gmsh_model(
                model,
                _tdim,
                # Globally refine
                number_options=number_options,
                string_options=string_options,
                view=kwds.get('view', False))

            mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

            gmsh.finalize()

            if return_lookup:
                return mesh, entity_functions, lookup
            return mesh, entity_functions
        return wrapper
    return mesh_it


def msh_gmsh_model(model, dim, number_options=None, string_options=None, view=False):
    '''Generate dim-D mesh of model according to options'''
    if number_options:
        for opt in number_options:
            gmsh.option.setNumber(opt, number_options[opt])

    if string_options:
        for opt in string_options:
            gmsh.option.setString(opt, string_options[opt])

    model.occ.synchronize()
    model.geo.synchronize()

    if view:
        gmsh.fltk.initialize()
        gmsh.fltk.run()
            
    model.mesh.generate(dim)

    nodes, data = get_nodes_data(model)

    return nodes, data


def get_nodes_data(model):
    '''Extract'''
    indices, nodes, _ = model.mesh.getNodes()
    indices -= 1
    nodes = nodes.reshape((-1, 3))

    physical_groups = model.getPhysicalGroups()
    data = {}
    for dim, tag in physical_groups:

        entities = model.getEntitiesForPhysicalGroup(dim, tag)
        # Grab entities of topological dimension which have the tag
        for entity in entities:
            element_data = model.mesh.getElements(dim, tag=entity)
            element_types, element_tags, node_tags = element_data
            # That enity is mesh by exatly one element type
            element_type, = element_types
            # The MSH type of the cells on the element
            num_el = len(element_tags[0])

            element_topology = node_tags[0].reshape((num_el, -1)) - 1
            cell_data = np.full(num_el, tag)
        
            if element_type in data:
                data[element_type]['topology'] = np.row_stack([
                    data[element_type]['topology'], element_topology
                ]
                )
                data[element_type]['cell_data'] = np.hstack([
                    data[element_type]['cell_data'], cell_data
                ]
            )
            else:
                data[element_type] = {'topology': element_topology,
                                      'cell_data': cell_data}

    # NOTE: gmsh nodes are ordered according to indices. The only control
    # we will have when making the mesh is that order in which we insert
    # the vertex is respected. So to make topology work we need to insert
    # them as gmsh would (according to indices)
    nodes = nodes[np.argsort(indices)]

    return nodes, data


def mesh_from_gmsh(nodes, element_data, TOL=1E-13):
    '''Return mesh and dict tdim -> MeshFunction over tdim-entities'''
    # We only support lines, triangles and tets
    assert set(element_data.keys()) <= set((1, 2, 4, 15)), tuple(element_data.keys())

    elm_tdim = {1: 1, 2: 2, 4: 3, 15: 0}    
    # The idea is first build the mesh (and cell function) and later
    # descent over entities to tag them
    cell_elm = max(element_data.keys(), key=lambda elm: elm_tdim[elm])  # 
    # Here are cells defined in terms of incident nodes (in gmsh numbering)
    cells_as_nodes = element_data[cell_elm]['topology']
    # The unique vertices make up mesh vertices
    vtx_idx = np.unique(cells_as_nodes)
    mesh_vertices = nodes[vtx_idx]  # Now we have our numbering
    # Want to make from old to new to redefine cells
    node_map = {old: new for (new, old) in enumerate(vtx_idx)}
    
    cells_as_nodes = np.fromiter((node_map[v] for v in cells_as_nodes.ravel()),
                                 dtype=cells_as_nodes.dtype).reshape(cells_as_nodes.shape)

    print(f'Mesh has {len(cells_as_nodes)} cells of type {cell_elm}.')
    # Cell-node-connectivity is enough to build mesh
    elm_name = {1: 'interval', 2: 'triangle', 3: 'tetrahedron'}
    cell_tdim = elm_tdim[cell_elm]
    # Since gmsh has nodes always as 3d we infer gdim from whether last
    # axis is 0
    if np.linalg.norm(mesh_vertices[:, 2]) < TOL:
        gdim = 2
        mesh_vertices = mesh_vertices[:, :gdim]
    else:
        gdim = 3

    cell = ufl.Cell(elm_name[cell_tdim], gdim)
    mesh = build_mesh(mesh_vertices, cells_as_nodes, cell)
    
    entity_functions = {}
    # Is there cell_function to build?
    if np.min(element_data[cell_elm]['cell_data']) > 0:
        f = df.MeshFunction('size_t', mesh, cell_tdim)
        f.array()[:] = element_data[cell_elm]['cell_data']
        entity_functions[cell_tdim] = f
    
    # The remainig entity functions need to looked up among entities of
    # the mesh
    for elm in element_data:
        # We dealt with cells already
        if elm == cell_elm:
            continue
        # All zeros is not interesting either
        if np.min(element_data[elm]['cell_data']) == 0:
            continue

        tdim = elm_tdim[elm]
        # Need maps
        if tdim > 0:
            # Same as with gmsh we want to encode entity in terms of its
            # vertex connectivity
            mesh.init(tdim)
            _, v2e = (mesh.init(0, tdim), mesh.topology()(0, tdim))
            _, e2v = (mesh.init(tdim, 0), mesh.topology()(tdim, 0))
        else:
            v2e, e2v = lambda x: (x, ), lambda x: (x, )

        f = df.MeshFunction('size_t', mesh, tdim, 0)

        for entity, tag in zip(element_data[elm]['topology'], element_data[elm]['cell_data']):
            # Encode in fenics
            entity = [node_map[v] for v in entity]
            # When we look at entities incide to the related vertices
            mesh_entity, = reduce(operator.and_, (set(v2e(v)) for v in entity))
            # ... there should be exactly one and
            assert set(entity) == set(e2v(mesh_entity))

            f[mesh_entity] = tag
            
        entity_functions[tdim] = f

    return mesh, entity_functions
