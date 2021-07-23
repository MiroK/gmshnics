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
