import glob, os
import cv2
from skimage import io
import dlib
from pathlib import Path
import numpy as np
from imutils import face_utils, resize
import xml.etree.cElementTree as ET

def glob_images(image_dir):
    images = sorted([_ for _ in glob.glob(os.path.join(image_dir, '*.png')) if '_seg' not in _])
    print(f'There are {len(images)} images.')
    return images

def glob_ldmks(image_dir):
    ldmks = sorted([_ for _ in glob.glob(os.path.join(image_dir, '*_ldmks.txt'))])
    print(f'There are {len(ldmks)} ldmk files.')
    return ldmks

def check_matching(images, ldmks):
    missing = []
    img_count = len(images)
    for i, img_name in enumerate(images):
        if "{}_ldmks.txt".format(img_name.split('.')[0]) not in ldmks:
            missing.append(img_name)
    print(f'{len(missing)}/{img_count} missing.')
    return [_ for _ in images if _ not in missing]

def bbox_from_dlib(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(gray, 1)
    bbox = []
    for face in faces:
        # (x, y, w, h)
        bbox.append(face_utils.rect_to_bb(face))
    return bbox

def bbox_from_ldmks(landmarks, offset=15):
    landmarks = np.array(landmarks)
    right, bottom = landmarks.max(axis=0)
    left, top = landmarks.min(axis=0)
    width = right - left 
    height = bottom - top
    # (x, y, w, h)
    return [(left - offset, top - offset, width + 2*offset , height + 2*offset)]

def read_landmarks(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        landmarks = []
        for l in lines:
            if l.strip():
                _x, _y = l.strip().split(' ')
                landmarks.append((int(float(_x)), int(float(_y))))
    assert len(landmarks) >= 68
    return landmarks

def createXMLSynthesis(images, option='dlib'):
    root_node = ET.Element('dataset')
    ET.SubElement(root_node, "name").text = "Microsoft Synthesis Face Dataset"
    ET.SubElement(root_node, "comment")
    images_node = ET.SubElement(root_node, "images")
    
    total = len(images)
    added = 0
    for icount, imagepath in enumerate(images):
        
        image = io.imread(imagepath)
        h, w, c = image.shape
        ldmkpath = "{}_ldmks.txt".format(imagepath.split('.')[0])
        landmarks = read_landmarks(ldmkpath) # [(x, y)]
         
        if option == 'dlib':
            crops = bbox_from_dlib(image) # (x, y, w, h)
        elif option == 'landmark':
            crops = bbox_from_ldmks(landmarks, 15)
        else:
            crops = bbox_from_dlib(image) or bbox_from_ldmks(landmarks, 15)        
        
        if not crops:
            continue
        
        crop = crops[0]
        
        image_node = ET.SubElement(images_node, "image")
        image_node.set("file", imagepath)
        image_node.set("width", "{:d}".format(w))
        image_node.set("height", "{:d}".format(h))
        
        box_node = ET.SubElement(image_node, "box")
        box_node.set("top", "{:d}".format(crop[1]))
        box_node.set("left", "{:d}".format(crop[0]))
        box_node.set("width", "{:d}".format(crop[2]))
        box_node.set("height", "{:d}".format(crop[3]))
        
        for i, landmark in enumerate(landmarks):
            part_node = ET.SubElement(box_node, "part")
            part_node.set('name', "{:02d}".format(i))
            part_node.set('x', "{:d}".format(landmark[0]))
            part_node.set('y', "{:d}".format(landmark[1]))        
        
        if (icount+1) % 100 == 0:
            print(f'processed {icount+1}/{total} images.')
            
        added += 1
        
    # return a tree
    count_node = ET.SubElement(root_node, "total")
    count_node.text = str(added)
    
    return ET.ElementTree(root_node)

def generate_xml_synthesis(root_dir, option='dlib'):
    print('GENERATING XML, OPTION=' + option)
#     root_dir = "/Users/dongxuhuang/Downloads/Landmark/dataset_5000"
    images = glob_images(root_dir)
    ldmks = glob_ldmks(root_dir)
    valided_images = check_matching(images, ldmks)
    
    print('building tree...')
    tree = createXMLSynthesis(valided_images, option)
    ET.indent(tree, space="\t", level=0)
    
    if option == 'dlib':
        output_name = 'labels_ms_synthesis_dlib.xml'
    elif option == 'landmark':
        output_name = 'labels_ms_synthesis_landmark.xml'
    else:
        output_name = 'labels_ms_synthesis_mix.xml'
        
    tree.write(output_name, encoding='utf-8', xml_declaration=True)

if __name__ == '__main__':
   root_dir = "/home/dha101/project/dataset/dataset_5000"
   generate_xml_synthesis(root_dir=root_dir, option='dlib')
   generate_xml_synthesis(root_dir=root_dir, option='landmark')
   generate_xml_synthesis(root_dir=root_dir, option='mixed')