import fileinput
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.applications import MobileNetV2
from keras.applications.mobilenetv2 import preprocess_input
from keras.layers import AveragePooling2D
from keras.models import Model
from keras.models import load_model
from res_manager import ResultManager

from src.agents import DQN_Agent
from src.cloud_apis import Baidu

plt.ion()

np.set_printoptions(precision=3)

np.random.seed(2)

with open('evaluation_results/image_reference_cache.defaultdict', 'rb') as f:
    ref_cache = pickle.load(f)


def plot_durations(y, title_list):
    plt.figure(2)
    plt.clf()
    plot_count = len(title_list)
    for idx, title in enumerate(title_list):
        plt.subplot(plot_count, 1, idx + 1)
        plt.plot(y[idx, :])
        plt.ylabel(title)
    plt.pause(0.0001)


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


def _gen_sample_set_imagenet(imagenet_train_path, samples_per_class):
    image_paths = []
    img_classes = os.listdir(imagenet_train_path)
    for img_class in img_classes:
        for image_name in np.random.choice(os.listdir("%s/%s" % (imagenet_train_path, img_class)),
                                           size=samples_per_class):
            sample_image_path = ("%s/%s/%s" % (imagenet_train_path, img_class, image_name))

            image_paths.append(sample_image_path)
    return image_paths


class RunningAgent(object):
    def __init__(self,
                 dqn_path,
                 banchmark_q,
                 cloud_backend,
                 recent_zone=10,
                 explor_rate=0.2,
                 acc_threshold=0.8,
                 reward_threshold=0.45):

        feature_extractor = MobileNetV2(include_top=False)
        x = feature_extractor.output
        x = AveragePooling2D(pool_size=(4, 4))(x)
        self.feature_extractor = Model(inputs=feature_extractor.input, outputs=x)

        self.agent = DQN_Agent(s_dim=1280,
                               a_dim=10,
                               epsilon_decay=0.99,
                               epsilon_min=0.02,
                               gamma=0.95,
                               replay_batchsize=256)

        self.agent.model = load_model(dqn_path)
        self.dqn_path = dqn_path
        self.agent.curr_exploration_rate = 0
        self.STATUS = "INFERENCE"  # INITIAL_TRAIN, INFERENCE, ESTIMATE, RETRAIN

        self.cloud_backend = cloud_backend
        self.banchmark_q = banchmark_q
        self.explor_rate = explor_rate
        self.recent_zone = recent_zone
        self.reward_threshold = reward_threshold
        self.acc_threshold = acc_threshold

        self.grad_scale_factor = 3

        self.agent_memory = defaultdict(list)
        self.running_log = defaultdict(list)
        self.last_env_step = None
        self.step_count = 0
        self.train_count = 0

    def infer(self, image):
        image_data = preprocess_input(np.expand_dims(np.asarray(image.resize((224, 224)), dtype=np.float32), axis=0))
        features = self.feature_extractor.predict(image_data)[0][0][0]
        state_action, action_id = self.agent.choose_action(features)
        return state_action, features, action_id, int([i for i in np.arange(5, 105, 10)][action_id])

    def remember(self, status, action, accuracy, ref_size, comp_size, upload_size, size_reward, reward,
                 image_path, recent_reward, recent_accuracy):
        self.running_log['status'].append(status)
        self.running_log['action'].append(action)
        self.running_log['accuracy'].append(1 if self.STATUS != "INFERENCE" else accuracy)
        self.running_log['agent_accuracy'].append(accuracy)
        self.running_log['ref_size'].append(ref_size)
        self.running_log['comp_size'].append(comp_size)
        self.running_log['upload_size'].append(upload_size)
        self.running_log['size_reward'].append(size_reward)
        self.running_log['reward'].append(reward)
        self.running_log['image_path'].append(image_path)
        self.running_log['step_count'].append(self.step_count)
        self.running_log['recent_reward'].append(recent_reward)
        self.running_log['recent_accuracy'].append(recent_accuracy)
        self.running_log['explor_rate'].append(self.explor_rate)
        self.running_log['agent_epsilon'].append(self.agent.curr_exploration_rate)

        if self.STATUS != "INFERENCE":  # in ESTIMATE, RETRAIN and INITIAL_TRAIN status, the agent can log everything
            self.agent_memory['image_path'].append(image_path)
            self.agent_memory['step_count'].append(self.step_count)
            self.agent_memory['accuracy'].append(accuracy)
            self.agent_memory['ref_size'].append(ref_size)
            self.agent_memory['comp_size'].append(comp_size)
            self.agent_memory['size_reward'].append(size_reward)
            self.agent_memory['reward'].append(reward)
            self.agent_memory['action'].append(action)
            self.agent_memory['recent_accuracy'].append(recent_accuracy)

    def agent_upload(self, image_path):
        image = Image.open(image_path).convert("RGB")
        self.step_count += 1
        state_action, features, action_id, action = self.infer(image)

        if ref_cache["%s##%s" % (image_path, action)] == {}:
            error_code, results, size = self.cloud_backend.recognize(image, action)
            ref_cache["%s##%s" % (image_path, action)] = {"error_code": error_code,
                                                          "results": results, "size": size,
                                                          "banchmark_q": action}
        else:
            cache = ref_cache["%s##%s" % (image_path, action)]
            error_code = cache['error_code']
            results = cache['results']
            size = cache['size']

        if error_code > 0: return 1, results

        # if status == "ESTIMATE"
        if ref_cache["%s##%s" % (image_path, self.banchmark_q)] == {}:
            error_code, ref_results, ref_size = self.cloud_backend.recognize(image, self.banchmark_q)
            ref_cache["%s##%s" % (image_path, self.banchmark_q)] = {"error_code": error_code, "results": ref_results,
                                                                    "size": ref_size, "banchmark_q": self.banchmark_q}
        else:
            cache = ref_cache["%s##%s" % (image_path, self.banchmark_q)]
            error_code = cache['error_code']
            ref_results = cache['results']
            ref_size = cache['size']

        if error_code > 0: return 2, ref_results

        ref_labels = np.array([line['keyword'] for line in ref_results])[np.argsort([line['score'] for line in ref_results])[::-1]][:1]
        accuracy = 1 if len(set(ref_labels) & set([line['keyword'] for line in results])) >= 1 else 0

        size_reward = size / ref_size
        reward = accuracy - size_reward

        recent_acc, recent_reward = self.estimate()


        # Remember current behavior
        log_dict = {"status": ["INITIAL_TRAIN", "INFERENCE", "ESTIMATE", "RETRAIN"].index(self.STATUS),
                    "accuracy": accuracy,
                    "ref_size": ref_size,
                    "comp_size": size,
                    "upload_size": ref_size + size if self.STATUS != "INFERENCE" else size,
                    "size_reward": size_reward,
                    "reward": reward,
                    "image_path": image_path,
                    "action": action,
                    "recent_accuracy": recent_acc,
                    "recent_reward": recent_reward
                    }
        self.remember(**log_dict)

        # Status drift
        if self.STATUS == "INFERENCE":
            self.explor_rate = np.clip(self.explor_rate, 0.2, 0.95)
            self.STATUS = "ESTIMATE" if np.random.uniform(low=0, high=1) < self.explor_rate else "INFERENCE"
        elif self.STATUS == "ESTIMATE":
            if len(self.agent_memory['recent_accuracy']) > self.recent_zone:
                grad = np.gradient(self.agent_memory['recent_accuracy'])[-1]
                self.explor_rate = self.explor_rate if np.isnan(grad) else self.explor_rate - self.grad_scale_factor * grad
                self.explor_rate = np.clip(self.explor_rate, 0.2, 0.95)

            if len(self.agent_memory['recent_accuracy']) > self.recent_zone and recent_acc < self.acc_threshold:
                self.agent.memory.clear()
                self.STATUS = "RETRAIN"
            else:
                self.STATUS = "ESTIMATE" if np.random.uniform(low=0, high=1) < self.explor_rate else "INFERENCE"
        elif self.STATUS == "RETRAIN":
            if self.last_env_step is not None:
                self.agent.remember(self.last_env_step['features'],
                                    self.last_env_step['action_id'],
                                    self.last_env_step['reward'],
                                    features)
            self.last_env_step = {"features": features, "action_id": action_id, "reward": reward}
            if recent_reward > self.reward_threshold and recent_acc > self.acc_threshold and self.agent.curr_exploration_rate < 0.2:
                self.STATUS = "INFERENCE"

                self.explor_rate = 0.2
                self.agent_memory = defaultdict(list)

                self.agent.model.save(self.dqn_path + ".retrain")
                self.agent.model = load_model(self.dqn_path + ".retrain")
                self.agent.curr_exploration_rate = 0
            else:
                self.train_count += 1
                if self.train_count > 128 and self.train_count % 5 == 0:
                    self.agent.learn()
                if self.train_count <= 128:
                    self.agent.curr_exploration_rate = 1  # exploration at the beginning steps

        return 0, log_dict

    def estimate(self):
        if len(self.agent_memory['reward']) < self.recent_zone:
            recent_reward = np.mean(self.agent_memory['reward'])
            recent_acc = np.mean(self.agent_memory['accuracy'])
        else:
            recent_reward = np.mean(self.agent_memory['reward'][-self.recent_zone:])
            recent_acc = np.mean(self.agent_memory['accuracy'][-self.recent_zone:])
        return recent_acc, recent_reward


#

if __name__ == '__main__':
    api = Baidu()
    rm = ResultManager('evaluation_results')

    running_agent = RunningAgent(dqn_path='evaluation_results/agent_DQN_train_baidu_DNIM.h5',
                                 banchmark_q=75,
                                 cloud_backend=api,
                                 )

    # imagenet_paths = _gen_sample_set_imagenet('/home/hsli/gnode02/imagenet-data/train/', 2)

    np.random.shuffle(subset)
    test_image_paths = subset + rm.load(7)['image_path'][:1200] + rm.load(7)['image_path']

    for idx, path in enumerate(test_image_paths):
        error_code, log_dict = running_agent.agent_upload(path)
        if error_code > 0: continue
        print(idx, end='\t\t')
        for k, v in log_dict.items():
            if k not in ['image_path', 'initial_reward', 'initial_action', 'ref_size', 'upload_size']:
                print("%s:%.2f" % (k, v), end='\t')
        print('\n')

        plot_keys = ['accuracy', 'size_reward', 'reward', 'action', 'recent_accuracy', 'upload_size', 'agent_epsilon', 'agent_accuracy', 'recent_reward', 'explor_rate']
        plot_durations(np.array([running_agent.running_log[key] for key in plot_keys]),
                       title_list=plot_keys)

        if idx % 5 == 0:
            plt.savefig('evaluation_results/running_retrain_fullDNIM.png', dpi=100)

            with open('evaluation_results/image_reference_cache.defaultdict', 'wb') as f:
                pickle.dump(ref_cache, f)

            rm.save(running_agent.running_log,
                    name='DNIM_agent_on_imagenet',
                    topic="AgentRetrain",
                    comment="retrain running log with 0.5 exploration rate",
                    replace_version='latest'
                    )
