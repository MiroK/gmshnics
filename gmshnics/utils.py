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
