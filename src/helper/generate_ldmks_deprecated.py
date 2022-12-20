import os, glob
import numpy as np
import cv2
from skimage import io
from pathlib import Path
from imutils import face_utils
import dlib

def match_ldmks_files(image_files, ldmks_files, LDMKS_FILE_REG):
    validated = []
    missing_ldmks = []
    for i, x in enumerate(image_files):
        name = x.split('.')[0]
        ldmk_file = LDMKS_FILE_REG.format(name)
        if ldmk_file in ldmks_files:
            validated.append(x)
        else:
            missing_ldmks.append(x)
        if i>0 and i % 1000 == 0:
            print(f'processede {i}/{len(images)}')
    
    return validated, missing_ldmks

def bbox_from_dlib(image):
    try:
        if image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        face_detector = dlib.get_frontal_face_detector()
        faces = face_detector(gray, 1)
    except Exception:
        raise Exception(f'failed to get bbox from dlib')
    
    if len(faces)>0: # assume only 1 face
        # (x, y, w, h)
        return face_utils.rect_to_bb(faces[0])
    else:
        return None
    
def bbox_from_ldmks(ldmk_file, image, offset=5):
    dim = image.shape
    with open(ldmk_file, 'r') as f:
        lines = f.readlines()
        x, y = [], []
        for i in lines:
            _x, _y = i.strip().split(' ')
            x.append(float(_x))
            y.append(float(_y))
        
        x_min = max(0, min(x) - offset)
        y_min = max(0, min(y) - offset)
        x_max = min(dim[0]-1, max(x) + offset)
        y_max = min(dim[1]-1, max(y) + offset)
        
        # (x, y, w, h)
        w = int(np.ceil(max(x_max - x_min, y_max - y_min)))
        return (int(x_min), int(y_min), w, w)

def get_bb_box(image_file, ldmk_file, offset=5, mode='mix'):
    try:
        image = io.imread(image_file)
    except Exception as e:
        raise Exception(f'failed to get bbox: cant\'t open {image_file}.')
    
    bbox = bbox_from_dlib(image)
    bbox2 = bbox_from_ldmks(ldmk_file, image, offset)
    
    if mode == 'mix':
        return bbox or bbox2
    elif mode == 'dlib':
        return bbox
    elif mode == 'ldmk':
        return bbox2
    else:
        raise ValueError('invalid bbox mode: mix, dlib, ldmk')

# get root path
root_dir = os.getcwd()
root_dir = str(Path(root_dir).parent.joinpath('dataset/dataset_1000'))
new_ldmks_dir = os.path.join(root_dir, 'generated_ldmks')
print(f'root path: {root_dir}')
print(f'new lmdk path: {new_ldmks_dir}')
LDMKS_FILE_REG = '{}_ldmks.txt'
if not os.path.isdir(new_ldmks_dir):
    os.mkdir(new_ldmks_dir)

# list all images in folder
images = sorted([_ for _ in glob.glob(os.path.join(root_dir, '*.png')) if '_seg' not in _])
print(f'There are {len(images)} images.')
pts_files = sorted([_ for _ in glob.glob(os.path.join(root_dir, '*_ldmks.txt'))])
print(f'There are {len(pts_files)} pts_files.')

validated, missing_ldmks = match_ldmks_files(images, pts_files, LDMKS_FILE_REG)
print(f'{len(validated)} validated, {len(missing_ldmks)} missing ldmks, {len(images)} total.')

for file in validated:
    name = file.split('.')[0]
    ldmk_file = LDMKS_FILE_REG.format(name)
    bbox = get_bb_box(file, ldmk_file, 8, mode='ldmk')
    with open(ldmk_file, 'r') as f:
        lines = [_ for _ in f.readlines() if _.strip() != '']
    
    filename = f"{name.split('/')[-1]}.txt"
    with open(os.path.join(new_ldmks_dir, filename), 'w') as f:
        f.write(' '.join([str(x) for x in bbox]) + '\n')
        f.write(''.join(lines))
    print(filename)