# encoding: utf-8
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from src.agents import DQN_Agent
from keras.applications import MobileNetV2
from keras.applications.mobilenetv2 import preprocess_input
from keras.layers import AveragePooling2D
from keras.models import load_model, Model
from res_manager import ResultManager

from src.cloud_apis import AmazonRekognition
from src.environment import EnvironmentAPI

plt.ion()

np.set_printoptions(precision=3)

tf.set_random_seed(2)
np.random.seed(2)

EVALUATION = False


def plot_durations(y):
    plt.figure(2)
    plt.clf()
    plt.subplot(511)
    plt.plot(y[:, 0])
    plt.ylabel('confidence')
    plt.subplot(512)
    plt.plot(y[:, 1])
    plt.ylabel('compression rate')
    plt.subplot(513)
    plt.plot(y[:, 2])
    plt.ylabel('reward')
    plt.subplot(514)
    plt.plot(y[:, 3])
    plt.ylabel('epsilon')
    plt.subplot(515)
    plt.plot(y[:, 4])
    plt.ylabel('action')
    plt.pause(0.0001)


if __name__ == '__main__':
    images_dir = '/home/hsli/gnode02/imagenet-data/train/'

    feature_extractor = MobileNetV2(include_top=False)
    x = feature_extractor.output
    x = AveragePooling2D(pool_size=(4, 4))(x)
    feature_extractor = Model(inputs=feature_extractor.input, outputs=x)

    rm = ResultManager('evaluation_results')
    agent_acc_size_dict = []
    origin_acc_size_dict = []

    agent = DQN_Agent(s_dim=1280,
                      a_dim=10,
                      epsilon_decay=0.99,
                      epsilon_min=0.02,
                      gamma=0.95,
                      replay_batchsize=256)

    if EVALUATION:
        agent.model = load_model('evaluation_results/agent_DQN_train_amazon_imagenet.h5')	# If in evaluation phase, replace this with the actual pretrained model
        agent.curr_exploration_rate = 0

    step_count = 0

    env = EnvironmentAPI(imagenet_train_path=images_dir,
                         cloud_agent=AmazonRekognition(),
                         dataset='imagenet',
                         cache_path='evaluation_results/image_reference_cache_amazon.defaultdict')
						 # In order to reduce some billing recognition service requests, I cached the recognized result locally. 
						 # Can be replaced by loading an empty dict from a pickled file. Navigate to the code for more details.

    train_log = defaultdict(list)
    plot_y = []
    plot_part = deque(maxlen=10)

    for i_episode in range(1):
        print("\n\nepisode %s:" % i_episode)
        image = env.reset()

        image_data = preprocess_input(np.expand_dims(np.asarray(image.resize((224, 224)), dtype=np.float32), axis=0))
        features = feature_extractor.predict(image_data)[0][0][0]

        while True:
            step_count += 1
            state_actions, action_id = agent.choose_action(features)
            action = [i for i in np.arange(5, 105, 10)][action_id]
            error_code, new_image, reward, done_flag, info = env.step(action)

            if error_code > 0:
                step_count -= 1
                print(error_code)
                continue

            train_log['image_path'].append(env.image_paths[env.curr_image_id])
            train_log['acc_r'].append(info['acc_r'])
            train_log['size_r'].append(info['size_r'])
            train_log['action'].append(action)
            train_log['reward'].append(reward)
            train_log['epsilon'].append(agent.curr_exploration_rate)

            print('\tstep %d\t' % step_count, end='\t')
            for k, v in info.items():
                print("%s: %.3f" % (k, v), end='\t')
            print('\n')

            if not done_flag:
                image_data = preprocess_input(
                    np.expand_dims(np.asarray(new_image.resize((224, 224)), dtype=np.float32), axis=0))
                new_features = feature_extractor.predict(image_data)[0][0][0]

                if not EVALUATION:
                    agent.remember(features, action_id, reward, new_features)
                    if 128 <= step_count <= 1600 and step_count % 5 == 0:
                        agent.learn()
                    if step_count <= 128:
                        agent.curr_exploration_rate = 1
            else:
                break

            # Plot
            plot_part.append(np.array([info['acc_r'], info['size_r'], reward, agent.curr_exploration_rate, action]))
            if step_count % 10 == 0:
                plot_y.append(np.mean(plot_part, axis=0))
                plot_durations(np.array(plot_y))
                env.update_cache('evaluation_results/image_reference_cache_amazon.defaultdict')
				# Update locally cached recognition results.

            features = new_features

            if step_count % 20 == 0:
                plt.savefig('evaluation_results/eval_amazon_imagenet.png', dpi=100)
				# In order to observe the running result, I saved the plotted data.

            if step_count % 200 == 0:
                rm.save(train_log,
                        name='eval_amazon_imagenet',
                        topic="AgentTrain",
                        comment="Train an agent on amazon and ImageNet dataset",
                        replace_version='latest'
                        )

                if not EVALUATION:
					# Update RL agent model
                    agent.model.save("evaluation_results/agent_DQN_train_amazon_imagenet.h5")

            if step_count >= 1300 and EVALUATION:
                break

