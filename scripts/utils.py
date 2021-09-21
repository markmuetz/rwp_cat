def get_cube(cubes, section, item):
    for cube in cubes:
        stash = cube.attributes['STASH']
        if stash.section == section and stash.item == item:
            return cube
    raise Exception('Cube with ({}, {}) not found'.format(key, value))

