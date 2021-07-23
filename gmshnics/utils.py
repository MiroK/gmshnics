import dolfin as df
import itertools


def first(iterable):
    '''[0]'''
    return next(iter(iterable))


def second(iterable):
    '''[1]'''
    return first(itertools.islice(iterable, 1, None))


def valid_key(name):
    '''whatever_dim'''
    dim = name.split('_')[-1]

    return name, int(dim)


def dump_h5(path, mesh, entity_foos=None):
    '''Store as HDF5File'''
    if not isinstance(entity_foos, dict):
        # Make up names if not given
        entity_foos = {f'entity_f_{f.dim()}': f for f in entity_foos}
    # The names have to conform to "whaterver_dim" template
    # So this runs or fails
    names, dims = zip(*map(valid_key, entity_foos))
    
    with df.HDF5File(mesh.mpi_comm(), path, 'w') as out:
        out.write(mesh, 'mesh')
        if entity_foos is not None:
            for name, f in entity_foos.items():
                out.write(f, name)
    return path


def load_h5(path, entity_f_names=None):
    '''Load mesh and entity functions (assuming dumped by `dump_h5`)'''
    if entity_f_names is None:
        try:
            import h5py
            entity_f_names = set(h5py.File(path, 'r').keys()) - set(('mesh', ))
        except ImportError:
            raise ValueError('missing h5py need entity_f_names')

    mesh = df.Mesh()
    
    with df.HDF5File(mesh.mpi_comm(), path, 'r') as h5:
        h5.read(mesh, 'mesh', False)
        
        entity_fs = {}
        for name in entity_f_names:
            # Extract to init right mesh_f
            _, dim = valid_key(name)
            if h5.has_dataset(name):
                entity_f = df.MeshFunction('size_t', mesh, dim)
                h5.read(entity_f, name)

                entity_fs[name] = entity_f

    return mesh, entity_fs


def set_facet_distance_field(sizes, facets, model, factory, tdim):
    '''
    Set mesh size specifying for each physical facet group the gmsh
    Threshold field. Here facets maps phys tag to list of facet indices
    '''
    assert all(v.keys() == set(('SizeMax', 'DistMax', 'SizeMin', 'DistMin'))
               for v in sizes.values())

    field = model.mesh.field
    factory.synchronize()
    
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
