import fileinput
import os
import pickle
import time
from collections import defaultdict

import tensorflow as tf
from PIL import Image

from src.imagenet import imagenet_label2class
from src.jpeg_utils import *

tf.set_random_seed(2)
np.random.seed(2)


def list_split(l, size):
    return [l[m:m + size] for m in range(0, len(l), size)]

class EnvironmentAPI(object):
    def __init__(self,
                 imagenet_train_path,
                 cloud_agent,
                 dataset,
                 cache_path,
                 sample_counts=2000,
                 reference_quality=75):

        self.imagenet_train_path = imagenet_train_path
        self.samples_per_class = int(sample_counts / 1000)
        self.cloud_agent = cloud_agent
        self.sample_counts=sample_counts
        self.dataset=dataset

        self.image_datalist = []
        self.label_datalist = []
        self.image_paths = []

        self.references = defaultdict(dict)
        self.ref_size_list = []
        self.ref_labels = []
        self.ref_confidences = []

        self.gen_ref(reference_quality)
        self.reset()
        self.load_cache(cache_path)

    def load_cache(self, cache_path):
        with open(cache_path, 'rb') as f:
            self.cache = pickle.load(f)

    def update_cache(self, cache_path):
        with open(cache_path, 'wb') as f:
            pickle.dump(self.cache, f)

    def _gen_sample_set_imagenet(self):
        image_paths = []
        image_labels = []

        img_classes = os.listdir(self.imagenet_train_path)
        for img_class in img_classes:
            for image_name in np.random.choice(os.listdir("%s/%s" % (self.imagenet_train_path, img_class)),
                                               size=self.samples_per_class):
                sample_image_path = ("%s/%s/%s" % (self.imagenet_train_path, img_class, image_name))
                image_label = imagenet_label2class[image_name.split('_')[0]]

                image_paths.append(sample_image_path)
                image_labels.append(image_label)
        return image_paths

    def _gen_sample_set_place365(self):
        with open('evaluation_results/places365_dataset.array', 'rb') as f:
            dataset = pickle.load(f)
        chosen_ids = np.random.choice([i for i in range(len(dataset))], self.sample_counts)
        image_paths = dataset[chosen_ids, 0]
        return image_paths

    def _gen_sample_set_DNIM(self):
        # Generate DNIM subset
        subset = []
        for time_stemp in os.listdir("/home/hsli/gnode02/DNIM/DNIM/time_stamp/"):
            for line in fileinput.input("/home/hsli/gnode02/DNIM/DNIM/time_stamp/" + time_stemp):
                name, date, hour, m = line.strip().split(" ")
                if 0 <= int(hour) < 6 or int(hour) > 19:
                    date = name.split("_")[0]
                    img_id = name.split("_")[1].split(".")[0]
                    subset.append("/home/hsli/gnode02/DNIM/DNIM/Image/sorted_by_time/%s_%s_%s.jpg" % (
                    date, img_id, time_stemp.split(".")[0]))
        return subset

    def data_initial(self):
        image_paths = self._gen_sample_set_imagenet()
        for image_path in image_paths:
            self.image_datalist.append(Image.open(image_path).convert("RGB"))
            self.image_paths.append(image_path)

    def gen_ref(self, ref_quality):
        if not os.path.exists("evaluation_results/%s_%s_ref%d.pkl" % (self.dataset, self.cloud_agent.api_name, self.sample_counts)):
            print("Reference not exists, generating...")

            self.data_initial()

            for idx, image in enumerate(self.image_datalist):
                img_path = self.image_paths[idx]
                time.sleep(0.1)
                if idx % 20 == 0:
                    print(".", end='')
                error_code, reg_results, ref_size = self.cloud_agent.recognize(image, ref_quality)
                if error_code == 0:
                    gt_id = np.argmax([line['score'] for line in reg_results])

                    self.references[img_path]['error_code'] = error_code
                    self.references[img_path]['ref_size'] = ref_size
                    self.references[img_path]['ref_label'] = reg_results[gt_id]['keyword']
                    self.references[img_path]['ref_confidence'] = reg_results[gt_id]['score']
                else:
                    self.references[img_path]['error_code'] = error_code
                    self.references[img_path]['error_msg'] = reg_results
                    self.references[img_path]['ref_size'] = ref_size

            with open("evaluation_results/%s_%s_ref%d.pkl" % (self.dataset, self.cloud_agent.api_name, self.sample_counts), 'wb') as f:
                pickle.dump(self.references, f)
            print("\nReference generated..")
        else:
            with open("evaluation_results/%s_%s_ref%d.pkl" % (self.dataset, self.cloud_agent.api_name, self.sample_counts), 'rb') as f:
                self.references = pickle.load(f)

            image_paths = self.references.keys()
            for image_path in image_paths:
                self.image_datalist.append(Image.open(image_path).convert("RGB"))
                self.image_paths.append(image_path)

            print("Reference loaded...")

    def reset(self):
        self.curr_image_id = 0
        return self.image_datalist[self.curr_image_id]

    def cloud_recognize(self, img_path, image, quality, gt_label, ref_confidence, ref_size):
        if self.cache["%s##%s" % (img_path, quality)] == {}:
            error_code, reg_results, size = self.cloud_agent.recognize(image, quality)
            self.cache["%s##%s" % (img_path, quality)] = {"error_code": error_code,
                                                          "results": reg_results,
                                                          "size": size,
                                                          "banchmark_q": quality}
        else:
            cache = self.cache["%s##%s" % (img_path, quality)]
            error_code = cache['error_code']
            reg_results = cache['results']
            size = cache['size']

        size_reward = size / ref_size
        if error_code == 0:
            if not gt_label in [line['keyword'] for line in reg_results]:
                return 0, 0, size_reward
            else:
                # reg_id = [line['keyword'] for line in reg_results].index(gt_label)
                # confidence = np.clip([line['score'] for line in reg_results][reg_id] / ref_confidence, 0.4, 1)
                # acc_reward = confidence

                return 0, 1, size_reward
        else:
            return 1, reg_results[0], 0

    def step(self, action):
        done_flag = False

        info = {}

        quality = int(action)

        path = self.image_paths[self.curr_image_id]
        reference = self.references[path]
        if reference['error_code'] == 0:
            error_code, acc_reward, size_reward = self.cloud_recognize(img_path=self.image_paths[self.curr_image_id],
                                                                       image=self.image_datalist[self.curr_image_id],
                                                                       quality=quality,
                                                                       gt_label=reference['ref_label'],
                                                                       ref_confidence=reference['ref_confidence'],
                                                                       ref_size=reference['ref_size']
                                                                       )
            if error_code == 0:

                reward = acc_reward - size_reward

                info['acc_r'] = acc_reward
                info['size_r'] = size_reward
                info['action'] = action
                info['reward'] = reward

                self.curr_image_id += 1

                if self.curr_image_id >= len(self.image_datalist):
                    done_flag = True
                    return 0, np.zeros(64), reward, done_flag, info

                features = self.image_datalist[self.curr_image_id]
                return 0, features, reward, done_flag, info

            else:
                self.curr_image_id += 1
                return 1, None, None, None, None
        else:
            self.curr_image_id += 1
            return 2, None, None, None, None


class RunningEnvironment(EnvironmentAPI):
    def __init__(self,
                 cloud_agent,
                 banchmark_quality):
        self.cloud_agent = cloud_agent
        self.banchmark_quality = banchmark_quality
