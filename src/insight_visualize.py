import cv2
import keras.backend as K
import numpy as np
from keras.applications import MobileNetV2
from keras.applications.mobilenetv2 import preprocess_input
from keras.layers import AveragePooling2D
from keras.models import load_model, Model
from keras.preprocessing import image


class Visualizer(object):
    def __init__(self, dqn_path):
        K.clear_session()
        feature_extractor = MobileNetV2(include_top=False, input_shape=((224, 224, 3)))
        dqn_x = feature_extractor.output
        dqn_x = AveragePooling2D(pool_size=(4, 4))(dqn_x)
        # feature_extractor = Model(inputs=feature_extractor.input, outputs=x)

        dqn = load_model(dqn_path)

        agent_out = dqn(dqn_x)

        self.model = Model(inputs=feature_extractor.input, outputs=agent_out)

    def visualize(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = self.model.predict(x)

        # Hotmap
        pred_id = np.argmax(preds[0])

        predicted_output = self.model.output[..., pred_id]
        last_conv_layer = self.model.get_layer('Conv_1')
        grads = K.gradients(predicted_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([self.model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x])
        for i in range(1280):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap /= np.max(heatmap)

        # Show on picture
        img = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img
        b, g, r = cv2.split(superimposed_img)
        superimposed_img = cv2.merge([r, g, b])

        return superimposed_img

    def format_img_float(self, float_img):
        return (float_img - np.min(float_img)) / (np.max(float_img) - np.min(float_img))

    def format_img_8bit(self, float_img):
        formatted_imposed = self.format_img_float(float_img) * 255
        return formatted_imposed.astype(np.uint8)


if __name__ == '__main__':
    visualizer = Visualizer('evaluation_results/agent_DQN_baidu_places365.h5')
    data = visualizer.visualize('/home/hsli/gnode02/scene_images/data_large/a/alley/00020767.jpg')
