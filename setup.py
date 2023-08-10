from setuptools import setup


setup(
    name='src',
    version='0.0.1',
    author='Nathan',
    description='small package for see who your face matches to',
    author_email='nategabrielk@icloud.com',
    packages=['src'],
    python_requires='>3.7',
    install_requires=[
        'mtcnn==0.1.0',
        'tensorflow==2.13.0',
        'keras==2.13.1',
        'keras-vggface==0.6',
        'keras_applications==1.0.8',
        'PyYAML',
        'tqdm',
        'scikit-learn',
        'streamlit',
        'bing-image-downloader'
    ]
)