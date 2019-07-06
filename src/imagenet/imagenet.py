import shutil
import numpy as np
from xml.dom.minidom import parse
import os
import json
import fileinput
from collections import defaultdict
from multiprocessing import Pool

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import to_categorical
from keras.metrics import top_k_categorical_accuracy

with open('imagenet/imagenet_class_index.json') as f:
    imagenet_dict = json.load(f)

imagenet_label2class = {}
imagenet_label2name = {}
imagenet_class2name = {}
imagenet_class2label = {}
for k, v in imagenet_dict.items():
    imagenet_label2class[v[0]] = int(k)
    imagenet_label2name[v[0]] = v[1]
    imagenet_class2name[int(k)] = v[1]
    imagenet_class2label[int(k)] = v[0]

# wordnet_dict = defaultdict(list)
# for line in fileinput.input('imagenet/wordnet.is_a.txt'):
#     a, b = line.split(' ')[:2]
#     wordnet_dict[a.strip()].append(b.strip())

# imagenet_label_name_list = np.array(imagenet_dict.values())[
#     np.array([int(item) for item in imagenet_dict.keys()]).argsort()]
# imagenet_labels = imagenet_label_name_list[:, 0]
# imagenet_names = imagenet_label_name_list[:, 1]


def xmlparser_ilsvrc15(xml_path):
    """
    Parse imagenet xml annotation files to get bounding boxes, object label and names etc.
    :param xml_path: path of the xml file.
    :return:
    """
    width, height, depth, xmin, xmax, ymin, ymax = [[], [], [], [], [], [], []]
    object_label_ids = []
    object_names = []
    object_labels = []
    DOMTree = parse(xml_path)
    data = DOMTree.documentElement
    parse_name = data.getElementsByTagName('filename')[0].firstChild.data.strip()
    size = data.getElementsByTagName('size')[0]
    width = int(size.getElementsByTagName('width')[0].firstChild.data)
    height = int(size.getElementsByTagName('height')[0].firstChild.data)
    # depth = int(size.getElementsByTagName('depth')[0].firstChild.data)
    object_count = len(data.getElementsByTagName('object'))

    objects = data.getElementsByTagName('object')
    for img_object in objects:
        object_labels += wordnet_dict[img_object.getElementsByTagName('name')[0].childNodes[0].data.strip()]

    object_labels = list(set(object_labels) & set(imagenet_labels))

    for object_label in object_labels:
        object_names.append(imagenet_names[np.where(imagenet_labels==object_label)])

    return parse_name, width, height, depth, xmin, xmax, ymin, ymax, object_count, object_labels, object_label_ids, object_names


def xmlparser(xml_path):
    """
    Parse imagenet xml annotation files to get bounding boxes, object label and names etc.
    :param xml_path: path of the xml file.
    :return:
    """
    width, height, depth, xmin, xmax, ymin, ymax = [[], [], [], [], [], [], []]
    object_label_ids = []
    object_names = []
    object_labels = []
    DOMTree = parse(xml_path)
    data = DOMTree.documentElement
    parse_name = data.getElementsByTagName('filename')[0].firstChild.data.strip()
    size = data.getElementsByTagName('size')[0]
    width = int(size.getElementsByTagName('width')[0].firstChild.data)
    height = int(size.getElementsByTagName('height')[0].firstChild.data)
    depth = int(size.getElementsByTagName('depth')[0].firstChild.data)
    object_count = len(data.getElementsByTagName('object'))

    objects = data.getElementsByTagName('object')
    for img_object in objects:
        object_labels.append(img_object.childNodes[1].firstChild.data.strip())
        bndbox = img_object.childNodes[9]
        xmin.append(int(bndbox.childNodes[1].firstChild.data))
        ymin.append(int(bndbox.childNodes[3].firstChild.data))
        xmax.append(int(bndbox.childNodes[5].firstChild.data))
        ymax.append(int(bndbox.childNodes[7].firstChild.data))

    for object_label in object_labels:
        object_index = imagenet_labels.tolist().index(object_label)
        object_label_ids.append(object_index)
        object_names.append(imagenet_names[object_index])

    return parse_name, width, height, depth, xmin, xmax, ymin, ymax, object_count, object_labels, object_label_ids, object_names


def generate_subset(source_path, target_path, subset_size):
    """
    Copy subset_size samples from imagenet_path to generated_path
    :param source_path: Imagenet root directory
    :param target_path: Target folder
    :param subset_size: How many samples to be copied.
    """

    if not os.path.exists("%s/Data/CLS-LOC/val/" % target_path):
        os.system('mkdir -p %s/Data/CLS-LOC/val/' % target_path)
    if not os.path.exists("%s/Annotations/CLS-LOC/val/" % target_path):
        os.system('mkdir -p %s/Annotations/CLS-LOC/val/' % target_path)

    for count, filename in enumerate(os.listdir("%s/Data/CLS-LOC/val/" % source_path)):
        if count > subset_size:
            break

        sample_name = filename.split('.')[0]
        shutil.copy("%s/Data/CLS-LOC/val/%s.JPEG" % (source_path, sample_name),
                    "%s/Data/CLS-LOC/val/%s.JPEG" % (target_path, sample_name))

        shutil.copy("%s/Annotations/CLS-LOC/val/%s.xml" % (source_path, sample_name),
                    "%s/Annotations/CLS-LOC/val/%s.xml" % (target_path, sample_name))

        if count % 100 == 0 and count > 0:
            print("%s samples copied..." % count)


def sample_parser(sample_name, imagenet_path="../Imagenet5W", target_size=(224, 224), max_objectcount=None,
                  select_classlabel=None):
    """
    Parse a sample name to generate sample batch. Useful to multiprocessing.
    :param sample_name: Sample name
    :param imagenet_path: Root folder of imagenet data set
    :param target_size: Size of image data, default to (224, 224)
    :param max_objectcount: Maximum number of objects present in ground true annotation.
            Default to None(all ground truth objects included)
    :param select_classlabel: class label selected.
    :return: (image data, annotation). Annotation in the format of np.array([0,0,0,...,1,0,0,0])
    """
    # Annotations
    parse_name, width, height, depth, xmin, xmax, ymin, ymax, object_count, object_labels, object_label_ids, object_names = xmlparser(
        "%s/Annotations/CLS-LOC/val/%s.xml" % (imagenet_path, sample_name))

    if select_classlabel is None or select_classlabel in object_labels:
        anno = np.clip(
            np.sum(to_categorical(object_label_ids[:max_objectcount], num_classes=1000), axis=0, dtype=np.uint8), 0,
            1)

        # Image data
        img = image.load_img("%s/Data/CLS-LOC/val/%s.JPEG" % (imagenet_path, sample_name), target_size=target_size)
        image_data = image.img_to_array(img)
        image_data = np.expand_dims(image_data, axis=0)
        # image_data = preprocess_input(image_data)

        return image_data, np.expand_dims(anno, axis=0)
    return None


def sample_parser_vid(image_path, xml_path, target_size=(224, 224),
                      max_objectcount=None, select_classlabel=None):
    """
    Parse a sample name to generate sample batch. Useful to multiprocessing.
    :param sample_name: Sample name
    :param imagenet_path: Root folder of imagenet data set
    :param target_size: Size of image data, default to (224, 224)
    :param max_objectcount: Maximum number of objects present in ground true annotation.
            Default to None(all ground truth objects included)
    :param select_classlabel: class label selected.
    :return: (image data, annotation). Annotation in the format of np.array([0,0,0,...,1,0,0,0])
    """
    # Annotations
    parse_name, width, height, depth, xmin, xmax, ymin, ymax, object_count, object_labels, object_label_ids, object_names = xmlparser(xml_path)

    if select_classlabel is None or select_classlabel in object_labels:
        anno = np.clip(
            np.sum(to_categorical(object_label_ids[:max_objectcount], num_classes=1000), axis=0, dtype=np.uint8), 0,
            1)

        # Image data
        img = image.load_img(image_path, target_size=target_size)
        image_data = image.img_to_array(img)
        image_data = np.expand_dims(image_data, axis=0)
        image_data = preprocess_input(image_data)

        return image_data, np.expand_dims(anno, axis=0)
    return None


def sample_generator(imagenet_path, target_size=(224, 224), max_objectcount=None, select_classlabel=None):
    """
    Generate one sample of (data, annotation) for each next() call.
    :param imagenet_path: Root folder of imagenet data set.
    :param target_size: Size of image data, default to (224, 224)
    :param max_objectcount: Maximum number of objects present in ground true annotation.
            Default to None(all ground truth objects included)
    :param select_classlabel: class label selected.
    :return: (image data, annotation). Annotation in the format of np.array([0,0,0,...,1,0,0,0])
    """
    count = 0
    while True:
        print("Imagenet data round %s" % count)
        for filename in os.listdir("%s/Data/CLS-LOC/val/" % imagenet_path):
            yield sample_parser(sample_name=filename.split('.')[0],
                                imagenet_path=imagenet_path,
                                target_size=target_size,
                                max_objectcount=max_objectcount,
                                select_classlabel=select_classlabel)
        count += 1


def top_k_acc(y_true, y_pred, k):
    """
    Calculate top K accuracy
    :param y_true: Ground truth labels.
    :param y_pred: Predictions.
    :param k: K
    :return: Top K accuracy.
    """
    sample_count = len(y_pred)
    hit_count = 0.
    for line_index, line_pred in enumerate(y_pred):
        line_true = y_true[line_index]
        top_k_pred_index = line_pred.argsort()[::-1][:k]
        for y_true_index in np.argwhere(line_true == 1):
            if y_true_index in top_k_pred_index:
                hit_count += 1.
                continue
    return hit_count / sample_count


def _generator_test():
    # Single thread generator test
    generator = sample_generator('../Imagenet5W')

    data, anno = generator.next()
    for i in range(500):
        _data, _anno = generator.next()
        data = np.vstack((data, _data))
        anno = np.vstack((anno, _anno))

    res50 = Net("ResNet50")
    predict_result = res50.full_inference(data)
    print("Kr: %s" % K.get_session().run(top_k_categorical_accuracy(anno, predict_result, k=5)))
    print("My: %s" % top_k_acc(anno, predict_result, 5))


def _multithread_parser_test():
    samplenames = [item.split('.')[0] for item in os.listdir('../Imagenet5W/Data/CLS-LOC/val/')]

    pool = Pool(processes=10)

    data_anno = pool.map(sample_parser, samplenames)

    data = np.vstack([item[0] for item in data_anno])
    anno = np.vstack([item[1] for item in data_anno])

    res50 = Net("ResNet50")
    predict_result = res50.full_inference(data)
    print("Kr: %s" % K.get_session().run(top_k_categorical_accuracy(anno, predict_result, k=5)))
    print("My: %s" % top_k_acc(anno, predict_result, 5))


if __name__ == '__main__':
    # _multithread_parser_test()
    # _generator_test()
    print(xmlparser('../000000.xml'))
