from gmshnics.utils import first, second
from itertools import count
import numpy as np
import tqdm

def pair_up(entities, x, factory, tol):
    '''Find entity matching x in center of mass'''
    found = False
    min_d = np.inf
    for entity in entities:
        dist = np.linalg.norm(factory.getCenterOfMass(*entity)[:len(x)] - x)
        found = dist < tol
        min_d = min(min_d, dist)
        if found:
            return entity
    assert found, f'Closest entity was {min_d} away'

    
def tag_matched_entities(coms, entities, model, factory, tagit, tol):
    '''Return tags for matches entities and leftovers'''
    edim, = set(map(first, entities))
    # Now the shape is going to claim it surfaces. If they are not made
    # specific by shape then all entities get one tag. Note that entities
    # [(edim, elementary tag)]
    if coms is None or not len(coms):
        tag = next(tagit)
        model.addPhysicalGroup(edim, list(map(second, entities)), tag)
        # We tagged it as one and exhausted
        return [(entities, tag)], []

    tags = []
    
    coms = list(reversed(coms))
    print(f'Matching {len(coms)} against {len(entities)}')
    pbar = tqdm.tqdm(len(coms))
    while coms:
        com = coms.pop()
        # Match remaining entities against center of mass
        entity = pair_up(entities, com, factory, tol)
        entities.remove(entity)

        tag = next(tagit)
        model.addPhysicalGroup(edim, [second(entity)], tag)
        factory.synchronize()
                
        tags.append((entity, tag))
        pbar.update(1)
    pbar.close()
    
    return tags, entities


def tag_volume_bdry(shape, volumes, model, factory):
    '''
    Compare shape sufaces in terms of coms with boundaries of the volumes
    dimTags making up the shape in the model
    '''
    shape_bdry = list(model.getBoundary(volumes))
    # Coordinates
    surf_coms = shape.com_surfaces
    # Count from 1
    return tag_matched_entities(surf_coms, shape_bdry, model, factory,
                                tagit=count(1), tol=1E-10)
