import numpy as np

def preprocess(data, preprocessor, **kwargs):
    if preprocessor == 'sqpatch':
        return sqpatch(data, patch_sz=kwargs['psz'])

def sqpatch(data, patch_sz):
    if len(data.shape) == 3:
        # MNIST-like
        assert data.shape[1] == data.shape[2], 'elements must be square'
        assert data.shape[1] % patch_sz == 0, ('element dimension must be ' +
            'an integer multiple of cropped dimension')

        n_patches = data.shape[1] // patch_sz
        patched_data = \
            np.array([square_crop(element, n_patches, patch_sz)
                for element in data], dtype=data.dtype)
        patched_data = np.array([patched_data.swapaxes(0, 1)])
        patched_data = patched_data.reshape(*patched_data.shape[:-2],
            patch_sz**2)
        return patched_data

    elif len(data.shape) == 4:
        # CIFAR-like
        pass

def square_crop(element, n_patches, patch_sz):
    cropped = np.empty((n_patches**2, patch_sz, patch_sz))
    for i in range(n_patches):
        for j in range(n_patches):
            cropped[i*n_patches+j] = \
                element[i*patch_sz:(i+1)*patch_sz, j*patch_sz:(j+1)*patch_sz]
    return cropped
