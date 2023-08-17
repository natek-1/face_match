import os
import argparse
import logging
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
from tqdm import tqdm


import pickle
from src.utils.all_utils import read_yaml, create_directory


logging_str = '[%(asctime)s: %(levelname)s: %(module)s]: %(message)s'
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'runnig_log.log'),
                     level=logging.INFO, format=logging_str, filemode='a')

def extractor(image_path, model, image_size=(224, 224)):
    img = image.load_img(image_path, target_size=image_size)
    img_array = image.img_to_array(img)
    expended_image = np.expand_dims(img_array, axis=0)
    preprocess_img = preprocess_input(expended_image)
    result = model.predict(preprocess_img).flatten()
    return result

def feature_extractor(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config['artifacts']
    artifacts_dir = artifacts['artifacts_dir']
    pickle_format_data_dir= artifacts['pickle_format_data_dir']
    img_pickle_file_name = artifacts['img_pickle_file_name']
    feature_extractor_dir = artifacts['feature_extraction_dir']
    extracted_features_name = artifacts['extracted_features_name']

    img_pickle_file_name = os.path.join(artifacts_dir, pickle_format_data_dir, img_pickle_file_name)

    filenames = pickle.load(open(img_pickle_file_name, 'rb'))
    model_name = params['base']['BASE_MODEL']
    include_tops = params['base']['include_top']
    pooling = params['base']['pooling']
    shape = params['base']['input_shape']

    model = VGGFace(model= model_name, include_top=include_tops, input_shape=shape, pooling=pooling)

    feature_extractor_path = os.path.join(artifacts_dir,feature_extractor_dir)
    create_directory(dirs=[feature_extractor_path])

    feature_name = os.path.join(feature_extractor_path , extracted_features_name)

    features = []

    for file in tqdm(filenames):
        features.append(extractor(file, model))
    logging.info('Feautures for all the image creates')
    pickle.dump(features, open(feature_name, 'wb'))
    logging.info(f'Feautures for all the image saves to {feature_name}')







if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', '-c', default='config/config.yaml')
    args.add_argument('--params', '-p', default='params.yaml')
    parsed_args = args.parse_args()

    try:
        logging.info(f'{">"*5} stage_02 started {"<"*5}')
        feature_extractor(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f'{">"*5} stage_02 completed {"<"*5}\n')
    except Exception as e:
        logging.exception(e)
        raise e
