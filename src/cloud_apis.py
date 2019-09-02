import urllib
import boto3
import time
from PIL import Image
import datetime
import json
import base64
from io import BytesIO
import Algorithmia


class Baidu(object):
    def __init__(self,
                 AK='gVdv****************NWA',
                 SK='OYa*********************HKc8'):
        self.api_name = "baidu"
        self.ak = AK
        self.sk = SK

        self.token = self._get_access_token()

    def _get_access_token(self):
        host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s' % \
               (self.ak, self.sk)
        request = urllib.request.Request(host)
        request.add_header('Content-Type', 'application/json; charset=UTF-8')
        response = urllib.request.urlopen(request)
        content = response.read().decode()
        return json.loads(content)['access_token']

    def _base64_encode(self, image, quality):
        f = BytesIO()
        image.save(f, format='jpeg', quality=quality)
        binary_data = f.getvalue()
        f.seek(0)
        size = len(f.getvalue())
        return base64.b64encode(binary_data), size

    def recognize(self, image, quality):
        time.sleep(0.1)
        img_b64, size = self._base64_encode(image, quality)
        request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"
        params = {"image": img_b64}
        params = urllib.parse.urlencode(params).encode(encoding='UTF8')

        access_token = self.token
        request_url = request_url + "?access_token=" + access_token
        request = urllib.request.Request(url=request_url, data=params)
        request.add_header('Content-Type', 'application/x-www-form-urlencoded')
        response = urllib.request.urlopen(request)
        content = response.read().decode()
        response_dict = json.loads(content)
        if not 'error_code' in response_dict.keys():
            return 0, response_dict['result'], size
        else:
            print(response_dict['error_msg'])
            return 1, [response_dict['error_msg']], size


class FacePP(Baidu):
    def __init__(self, AK='kf***********************6Mr3', SK='ta0*********************DLP'):
        self.api_name = "face_plusplus"
        self.ak = AK
        self.sk = SK

    def recognize(self, image, quality):
        time.sleep(0.1)
        img_b64, size = self._base64_encode(image, quality)
        request_url = "https://api-cn.faceplusplus.com/imagepp/beta/detectsceneandobject"
        params = {"api_key": self.ak,
                  "api_secret": self.sk,
                  "image_base64": img_b64}
        params = urllib.parse.urlencode(params).encode(encoding="UTF-8")
        request = urllib.request.Request(url=request_url, data=params)

        try:
            response = urllib.request.urlopen(request)
            content = response.read().decode()
            response_dict = json.loads(content)

            if not "error_message" in response_dict.keys():
                # print(response_dict['time_used'])
                if len(response_dict['objects']) == 0:
                    return 2, [{"keyword": "", "score": 1e-6}], size
                result_dicts = [{"keyword": line_dict['value'], "score": line_dict['confidence'] / 100.} for line_dict
                                in response_dict['objects']]
                return 0, result_dicts, size
            else:
                print(response_dict['error_message'])
                return 1, [response_dict['error_message']], size
        except Exception as e:
            print(e)
            return 3, [{"keyword": "", "score": 1e-6}], size


class AlgorithmiaAPI(Baidu):
    def __init__(self, AK='sim************************nEB61'):
        self.client = Algorithmia.client(AK)
        self.api_name = "Algorithmia"

    def upload(self, image, quality):
        file_location = "data://hosea1008/classification_images/%s.jpg" % datetime.datetime.now().strftime(
            '%Y-%m-%d_%H:%M:%S.%f')
        datafile = self.client.file(file_location)

        f = BytesIO()
        image.save(f, format='jpeg', quality=quality)
        size = len(f.getvalue())
        f.seek(0)

        datafile.client.putHelper(datafile.url, f)
        return file_location, size

    def recognize(self, image, quality):
        file_location, size = self.upload(image, quality)
        algo = self.client.algo('yavuzkomecoglu/ImageClassification/0.1.0')
        try:
            result = algo.pipe({"image": file_location}).result
            if len(result['predictions']) == 0:
                return 2, [{"keyword": "", "score": 1e-6}], size
            else:
                return 0, [{"keyword": line_dict['label'], "score": line_dict['probability']} for line_dict in
                           result['predictions']], size
        except Exception as e:
            print(e)
            return 1, [e], size


class AmazonRekognition(Baidu):
    def __init__(self):
        self.s3 = boto3.client('s3',
                               aws_access_key_id='AKIA*************672Q',
                               aws_secret_access_key='65Y**************************tMr')

        self.recognizer = boto3.client('rekognition',
                                       aws_access_key_id='AKIA**************672Q',
                                       aws_secret_access_key='65Y*************************tMr',
                                       region_name='ap-northeast-2'
                                       )
        self.bucket_name = 'fastinference-images-korea'
        self.api_name = 'amazon'

    def upload(self, image, quality):
        f = BytesIO()
        image.save(f, format='jpeg', quality=quality)
        size = len(f.getvalue())
        f.seek(0)

        self.s3.put_object(
            Bucket=self.bucket_name,
            Key='test5.jpeg',
            Body=f,
            ContentType='image/jpeg'
        )
        return size

    def recognize(self, image, quality):
        size = self.upload(image, quality)
        try:
            result = self.recognizer.detect_labels(
                Image={'S3Object': {'Bucket': self.bucket_name, 'Name': 'test5.jpeg'}}, MaxLabels=10)
            if len(result['Labels']) == 0:
                return 2, [{"keyword": "", "score": 1e-6}], size
            else:
                return 0, [{"keyword": line_dict['Name'], "score": line_dict['Confidence'] / 100.} for line_dict in
                           result['Labels']], size
        except Exception as e:
            print(e)
            return 1, [e], size


if __name__ == '__main__':
    amazon = AmazonRekognition()
    image = Image.open('/home/hsli/imagenet-data/train/n03085013/n03085013_773.JPEG')
    amazon.recognize(image, 75)

