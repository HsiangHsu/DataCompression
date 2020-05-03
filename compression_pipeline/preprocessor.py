import numpy as np

def preprocess(data, preprocessor, **kwargs):
    if preprocessor == 'sqpatch':
        return sqpatch(data, patch_sz=kwargs['psz'])

def sqpatch(data, patch_sz):
    # if len(data.shape) == 3:
    #     # MNIST-like
    #     assert data.shape[1] == data.shape[2], 'elements must be square'
    #     assert data.shape[1] % patch_sz == 0, ('element dimension must be ' +
    #         'an integer multiple of cropped dimension')

    #     patches_per_side = data.shape[1] // patch_sz
    #     patched_data = \
    #         np.array([square_crop(element, patches_per_side, patch_sz)
    #             for element in data], dtype=data.dtype)
    #     patched_data = np.array([patched_data.swapaxes(0, 1)])
    #     patched_data = patched_data.reshape(*patched_data.shape[:-2],
    #         patch_sz**2)
    #     return patched_data

    assert data.shape[-1] == data.shape[-2], 'elements must be square'
    assert data.shape[-1] % patch_sz == 0, ('element dimension must be ' +
        'an integer multiple of cropped dimension')
    patches_per_side = data.shape[-1] // patch_sz
    n_patches = patches_per_side**2

    if len(data.shape) == 3:
        data = data.reshape(data.shape[0], 1, data.shape[1], data.shape[2])
    data = data.swapaxes(0, 1)

    n_layers = data.shape[0]
    n_elements = data.shape[1]

    patched_data = np.empty((n_layers, n_elements, n_patches,
        patch_sz, patch_sz), dtype=data.dtype)

    for i in range(n_layers):
        for j in range(n_elements):
                patched_data[i][j] = square_crop(data[i][j],
                    patches_per_side, patch_sz)

    patched_data = patched_data.swapaxes(1,2)
    patched_data = patched_data.reshape(*patched_data.shape[:-2],
        patch_sz**2)
    return patched_data

def square_crop(element, patches_per_side, patch_sz):
    cropped = np.empty((patches_per_side**2, patch_sz, patch_sz))
    for i in range(patches_per_side):
        for j in range(patches_per_side):
            cropped[i*patches_per_side+j] = \
                element[i*patch_sz:(i+1)*patch_sz, j*patch_sz:(j+1)*patch_sz]
    return cropped
