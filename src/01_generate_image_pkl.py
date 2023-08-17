import os
import argparse
import logging
import pickle
from src.utils.all_utils import read_yaml, create_directory

logging_str = '[%(asctime)s: %(levelname)s: %(module)s]: %(message)s'
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'runnig_log.log'),
                     level=logging.INFO, format=logging_str, filemode='a')

def generate_data_pickfle_file(config_path: str, params_path):
    # Reading information from out config 
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    #Gathering information about those config 
    artifacts = config['artifacts']
    artifacts_dir = artifacts['artifacts_dir']
    pickle_format_data_dir= artifacts['pickle_format_data_dir']
    img_pickle_file_name = artifacts['img_pickle_file_name']
    

    ## Setting up folder where our images will be for easy access
    raw_local_dir_path = os.path.join(artifacts_dir, pickle_format_data_dir)
    create_directory(dirs=[raw_local_dir_path])


    pickle_file = os.path.join(raw_local_dir_path, img_pickle_file_name)
    data_path = params['base']['data_path']

    actors = os.listdir(data_path)

    # Getting the list of all the files with out data 
    filenames = []
    for actor in actors:
        actor_path = os.path.join(data_path, actor)
        for file in os.listdir(actor_path):
            filenames.append(os.path.join(actor_path, file))
    logging.info(f'Total number of actors are: {len(actors)}')
    logging.info(f'Total number of images are: {len(filenames)}')

    pickle.dump(filenames, open(pickle_file, 'wb'))



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', '-c', default='config/config.yaml')
    args.add_argument('--params', '-p', default='params.yaml')
    parsed_args = args.parse_args()

    try:
        logging.info(f'{">"*5} stage_01 started {"<"*5}')
        generate_data_pickfle_file(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f'{">"*5} stage_01 completed {"<"*5}\n')
    
    except Exception as e:
        logging.exception(e)
        raise e

