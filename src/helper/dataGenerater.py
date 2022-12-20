import random
import os
import xml.etree.cElementTree as ET

def loadXML(xmlpath: str, data_dir: str, limit: int=1000) -> dict:
    xmlpath = os.path.join(data_dir, xmlpath)
    tree = ET.parse(xmlpath).getroot()
    images = [elem for elem in tree.iter('image')][:int(limit)]
    random.shuffle(images)
    
    loaded = []
    for image in images:
        file = image.attrib['file']
        # clean absolute path
        if '/home/dha101/sfuhome/project/dataset/' in file:
            file = file.replace('/home/dha101/sfuhome/project/dataset/', '')
        else:
            file = os.path.join('ibug_300W_large_face_landmark_dataset/', file)

        file = os.path.join(data_dir, file)
        box = image.find('box').attrib
        (top, left, width, height) = (int(box['top']), int(box['left']), 
                                      int(box['width']), int(box['height']))
        landmarks = []
        for part in image.find('box').findall('part'):
            landmarks.append((int(part.attrib['x']), int(part.attrib['y'])))
        loaded.append({
            "file": file,
            "box": (left, top, width, height), # (x, y, w, h)
            "landmarks": landmarks[:68] # [(x, y), (x, y)]
        })
    return loaded

def generate_train_data(data_dir, option, size=1000, ft_ratio=0.8):
    # Source dataset:
    # 300W train: 6666
    # dlib: 2855
    # other: 5000   

    fake_count = int(size * ft_ratio)
    true_count = size - fake_count

    ibug_label = "labels_ibug_300W_train.xml"
    dlib_label = "labels_ms_synthesis_dlib.xml"
    ldmk_label = "labels_ms_synthesis_landmark.xml"
    mix_label = "labels_ms_synthesis_mix.xml"
    
    print(f'Dataset size: {size}, synthesis: {fake_count}, real: {true_count}')
    print(f'Bbox option: {option}')
    
    if fake_count > 2855 and option == "dlib":
        print('request dlib generated bbox but not enough images. Use mix option')
        option = 'mix'
    
    if fake_count > 5000 or true_count > 6666:
        raise Exception('There are only 5000 synthesis images and 6666 true images')
    
    if option == 'dlib':
        fake_data = loadXML(dlib_label, data_dir, limit=fake_count)
    elif option == 'landmark':
        fake_data = loadXML(ldmk_label, data_dir, limit=fake_count)
    else:
        fake_data = loadXML(mix_label, data_dir, limit=fake_count)
        
    if true_count > 0:
        true_data = loadXML(ibug_label, data_dir, limit=true_count)
        
        result = fake_data + true_data
        random.shuffle(result)
    else:
        result = fake_data
        
    return result
