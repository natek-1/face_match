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
        'PyYAML==6.0.1',
        'tqdm==4.66.1',
        'scikit-learn==1.3.0',
        'streamlit==1.25.0',
        'bing-image-downloader',
        'numpy==1.24.3'
    ]
)
