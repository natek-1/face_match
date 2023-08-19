from src.utils.all_utils import read_yaml, create_directory
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow .keras.applications import resnet50
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np

CONFIG_PATH = 'config/config.yaml'
PARAMS_PATH = 'params.yaml'
config = read_yaml(CONFIG_PATH)
params = read_yaml(PARAMS_PATH)

artifacts = config['artifacts']
artifacts_dir = artifacts['artifacts_dir']

#upload (inputs)
upload_image_dir = artifacts['upload_image_dir']
upload_path = os.path.join(artifacts_dir, upload_image_dir)

#pickle format data dir
pickle_format_data_dir= artifacts['pickle_format_data_dir']
img_pickle_file_name = artifacts['img_pickle_file_name']

raw_local_dir_path = os.path.join(artifacts_dir, pickle_format_data_dir)
pickle_file = os.path.join(raw_local_dir_path, img_pickle_file_name)

# feauture path
pickle_file = os.path.join(raw_local_dir_path, img_pickle_file_name)
feature_extractor_dir = artifacts['feature_extraction_dir']
extracted_features_name = artifacts['extracted_features_name']
feature_extractor_path = os.path.join(artifacts_dir,feature_extractor_dir)
feature_name = os.path.join(feature_extractor_path , extracted_features_name)

include_tops = params['base']['include_top']
pooling = params['base']['pooling']
shape = params['base']['input_shape']

detector = MTCNN()
model = resnet50.ResNet50(include_top=include_tops,input_shape=(224,224,3),pooling=pooling)
feature_list = pickle.load(open(feature_name,'rb'))
filenames = pickle.load(open(pickle_file,'rb'))

# save the uploaded image
def save_upload_image(upload_image, upload_path=upload_path):
    try:
        create_directory(dirs=[upload_path])

        with open(os.path.join(upload_path, upload_image.name), 'wb') as image:
            image.write(upload_image.getbuffer())
        return True
    except Exception as e:
        return False
    
# extract_features
def extract_features(img_path,model=model,detector=detector):
    img = cv2.imread(img_path)
    result = detector.detect_faces(img)
    if result != []:
        x, y, width, height =  result[0]['box']
        face = img[y: y+height, x:x+width]

        st.image(face)
        print
    
        #extract feature 
        image = Image.fromarray(face)
        image = image.resize(224, 224)
        face_array = np.asarray(image)
        face_array = face_array.astype('float32')
        expended_image = np.expand_dims(face_array, axis=0)
        preprocess_img = resnet50.preprocess_input(expended_image)
        result = model.predict(preprocess_img).flatten()

    return result

def recommend(feature_list : list(np.array), feature: np.array):
    similarity = []
    for feauture_y in feature_list:
        similarity.append(cosine_similarity(feature.reshape(1, -1), feauture_y.reshape(1, -1))[0][0])
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]

    return index_pos



# streamlit
st.title('Face matching application')

upload_image = st.file_uploader('choose an image', type=['png', 'jpg', 'jpeg'])

if upload_image is not None:
    print(upload_image)
    if save_upload_image(upload_image ):
        display_image = Image.open(upload_image)

        # Extract Feature
        feature = extract_features(os.path.join(upload_path, upload_image.name))

        # make selection
        if feature != []:
            index  = recommend(feature_list, feature)
        
