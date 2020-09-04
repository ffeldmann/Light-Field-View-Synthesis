import os
import numpy as np
import scipy.misc

data_source = '/home/tatev/uni/7.semester/project/data/LF_data/full_data/'
save_numpy = '/home/tatev/uni/7.semester/project/data/data_as_numpy/'
numpy_source = save_numpy
test = ['bedroom', 'bicycle', 'herbs', 'origami']
# train = []
def crop(image):
    '''
    Takes an array image with shape (512,512,3) and crops
    it to patches (48,48,3) as described in paper
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Alperovich_Light_Field_Intrinsics_CVPR_2018_paper.pdf
    '''
    px = 48
    py = 48
    sx = 16
    sy = 16
    ngrid = 32
    patches = np.zeros([14,14,48,48,3])
    for x in range(0,14):
        for y in range(0,14):
            upper_left_x = sx + x * ngrid
            upper_left_y = sy + y * ngrid
            patches[x][y] = image[upper_left_x:upper_left_x+px, upper_left_y:upper_left_y+py]

    return patches

def save_data():
    '''
    RUN in PYTHON2 in provided tool
    Saves the LF data in npy files for horizontal
    and vertical views for needed crossing epipolar volumes
    '''
    data_folders = os.listdir(data_source)
    for lf_folder in data_folders:
        save_dir = os.path.join(save_numpy, lf_folder)
        os.makedirs(save_dir)
        hor = os.path.join(save_dir, 'h.npy')
        vert = os.path.join(save_dir, 'v.npy')

        data_folder = os.path.join(data_source, lf_folder)
        import file_io
        LF = file_io.read_lightfield(data_folder)
        LF = LF.astype(np.float32)/255.0
        np.save(hor, LF[4])
        np.save(vert, LF[:,4,:,:,:])

def save_patches_as_image(patches, path, k, idx):
    '''
    Takes a croped patches 14x14x48x48x3 and saves as
    separate images with names 0 to 195
    '''
    for i in range(14):
        for j in range(14):
            file = os.path.join(path, str(idx*196+i+j*14), "{}.png".format(k))
            scipy.misc.imsave(file, patches[i][j])

# def create_data():
#     save_h = '/home/tatev/uni/7.semester/project/data/h'
#     save_v = '/home/tatev/uni/7.semester/project/data/v'
#     data_folders = os.listdir(numpy_source)
#     for lf_folder in data_folders:
#         h_folder = os.path.join(save_h, lf_folder)
#         v_folder = os.path.join(save_v, lf_folder)
#         with open(os.path.join(numpy_source,lf_folder,'h.npy'), 'rb') as f:
#             hor = np.load(f)
#         with open(os.path.join(numpy_source,lf_folder,'v.npy'), 'rb') as f:
#             vert = np.load(f)
#         for k in range(9):
#             h_save_dir = os.path.join(h_folder, str(k))
#             v_save_dir = os.path.join(v_folder, str(k))
#             os.makedirs(h_save_dir)
#             os.makedirs(v_save_dir)
#             h_patches = crop(hor[k])
#             v_patches = crop(vert[k])
#             save_patches_as_image(h_patches, h_save_dir)
#             save_patches_as_image(v_patches, v_save_dir)

def create_data():
    save_h = '/home/tatev/uni/7.semester/project/data/h'
    save_v = '/home/tatev/uni/7.semester/project/data/v'
    data_folders = os.listdir(numpy_source)
    for lf_folder in data_folders:
        h_folder = os.path.join(save_h, lf_folder)
        v_folder = os.path.join(save_v, lf_folder)
        os.makedirs(h_folder)
        os.makedirs(v_folder)
        for k in range(196):
            h_save_dir = os.path.join(h_folder, str(k))
            v_save_dir = os.path.join(v_folder, str(k))
            os.makedirs(h_save_dir)
            os.makedirs(v_save_dir)
        with open(os.path.join(numpy_source,lf_folder,'h.npy'), 'rb') as f:
            hor = np.load(f)
        with open(os.path.join(numpy_source,lf_folder,'v.npy'), 'rb') as f:
            vert = np.load(f)
        for k in range(9):
            h_patches = crop(hor[k])
            v_patches = crop(vert[k])
            save_patches_as_image(h_patches, h_folder, k)
            save_patches_as_image(v_patches, v_folder, k)

def create_data_enumerate():
    save_h = '/home/tatev/uni/7.semester/project/data/hh'
    save_v = '/home/tatev/uni/7.semester/project/data/vv'
    data_folders = os.listdir(numpy_source)
    for idx in range(28*196):
        h_folder = os.path.join(save_h, str(idx))
        v_folder = os.path.join(save_v, str(idx))
        os.makedirs(h_folder)
        os.makedirs(v_folder)
    for idx, lf_folder in enumerate(data_folders):
        for k in range(196):
            h_save_dir = os.path.join(save_h, str(idx*196+k))
            v_save_dir = os.path.join(save_v, str(idx*196+k))

        with open(os.path.join(numpy_source,lf_folder,'h.npy'), 'rb') as f:
            hor = np.load(f)
        with open(os.path.join(numpy_source,lf_folder,'v.npy'), 'rb') as f:
            vert = np.load(f)
        for k in range(9):
            h_patches = crop(hor[k])
            v_patches = crop(vert[k])
            save_patches_as_image(h_patches, save_h, k, idx)
            save_patches_as_image(v_patches, save_v, k, idx)


def create_data_splited():
    save_h = '/home/tatev/uni/7.semester/project/data/data_folder/train/h'
    save_v = '/home/tatev/uni/7.semester/project/data/data_folder/train/v'
    data_folders = os.listdir('/home/tatev/uni/7.semester/project/data/LF_data _split/train/')
    for idx in range(24*196):
        h_folder = os.path.join(save_h, str(idx))
        v_folder = os.path.join(save_v, str(idx))
        os.makedirs(h_folder)
        os.makedirs(v_folder)
    for idx, lf_folder in enumerate(data_folders):
        for k in range(196):
            h_save_dir = os.path.join(save_h, str(idx*196+k))
            v_save_dir = os.path.join(save_v, str(idx*196+k))

        with open(os.path.join(numpy_source,lf_folder,'h.npy'), 'rb') as f:
            hor = np.load(f)
        with open(os.path.join(numpy_source,lf_folder,'v.npy'), 'rb') as f:
            vert = np.load(f)
        for k in range(9):
            h_patches = crop(hor[k])
            v_patches = crop(vert[k])
            save_patches_as_image(h_patches, save_h, k, idx)
            save_patches_as_image(v_patches, save_v, k, idx)

