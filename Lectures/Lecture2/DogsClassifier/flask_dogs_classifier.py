
import base64
import io
import os
import json
import torch 

from flask import Flask, jsonify, abort, request, make_response, url_for
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224
MODEL_RESOURCES_DIR = 'models'

class DogsClassifier:


    def __init__(self, model_type, model_dict, index_mapper):
        self._index_mapper = index_mapper
        self._model = self._create_model_from(model_dict)
        self._model_transform = self._create_model_transform(model_type)

    @property
    def available_classes(self):
        return list(self._index_mapper.values())

    @staticmethod
    def _create_model_from(model_dict):
        model = model_dict['model']
        model.load_state_dict(model_dict['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False

        model.eval()
        return model
        
    @staticmethod
    def _create_model_transform(model_type):
        if model_type == 'resnet':
            return transforms.Compose(
                [ transforms.Resize(IMAGE_SIZE),
                  transforms.CenterCrop(IMAGE_SIZE),
                  transforms.ToTensor(),
                  transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]
                    )
                ]
            )
                        
        raise KeyError(f"{model_type} is not supported")


    @classmethod
    def create_from(cls, model_directory) -> 'DogsClassifier':
        
         model_dict = torch.load(os.path.join(model_directory, 'finetuned_model.pth'))
         with open(os.path.join(model_directory, 'breeds_mapping.json')) as fin:
            index_mapper = json.load(fin) 
            index_mapper = {int(key): val for key, val in index_mapper.items()}
             
            return cls('resnet', model_dict, index_mapper)


    def __call__(self, image):
        
        transformed_image = self._model_transform(image)
        model_input = transformed_image[None, ...]
        outputs = self._model(model_input)
        _, preds = torch.max(outputs.data, 1)
        print(self._index_mapper, preds[0], int(preds[0]))
        breed_str = self._index_mapper[int(preds[0])]

        return breed_str
    

from flask import Flask, jsonify, abort, request, make_response, url_for

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


@app.route('/breeds', methods=['GET'])
def get_breeds():
    return app.model.available_classes


@app.route('/predict', methods=['POST'])
def make_prediction():
    if 'encoded_image' not in request.json:
        abort(400, 'Not found image')
        
    encoded_image = request.json['encoded_image']
    image_bytes = base64.b64decode(encoded_image)
    image = Image.open(io.BytesIO(image_bytes))

    dog_breed = app.model(image)
    return {'breed': dog_breed}


if __name__ == '__main__':
    app.model = DogsClassifier.create_from(MODEL_RESOURCES_DIR)
    app.run(host="0.0.0.0", port=5000)
