#! /usr/bin/env python
import json
import sys
import re
# Dependencies can be installed by running:
# pip install keras tensorflow h5py pillow

# Run script as:
# ./extract-features.py images/*.jpg

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

def extract_class(filename):
    """
    Extracting classname based on filename 
    input : ./images/german_shorthaired_59.jpg
    output : german_shorthaired
    """
    return re.sub(r'[0-9]', '', filename.split('/')[-1])[:-5].strip('_')

def extract_index(filename):
    """
    Extracting index based on filename 
    input : ./images/german_shorthaired_59.jpg
    output : 59
    """
    return int(filename.split('/')[-1].split('.')[0].split('_')[-1])

def main():
    # Load model VGG16 as described in https://arxiv.org/abs/1409.1556
    # This is going to take some time...
    base_model = VGG16(weights='imagenet')
    # Model will produce the output of the 'fc2'layer which is the penultimate neural network layer
    # (see the paper above for mode details)
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)

    # For each image, extract the representation
    for image_path in sys.argv[1:]:
        features = extract_features(model, image_path)
        #add index and class at the e begining of the list
        features.insert(0,extract_index(image_path))
        features.insert(0,extract_class(image_path))
 

        with open(image_path + ".json", "w") as out:
            out.write(json.dumps(features).strip('[]').replace('"',''))

def extract_features(model, image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    return features.tolist()[0]

if __name__ == "__main__":
    main()