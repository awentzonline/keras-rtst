#!/usr/bin/env python
from distutils.core import setup
from setuptools import find_packages

setup(
    name='keras-rtst',
    version='0.0.9',
    description='An implementation of real-time style transfer in Keras',
    author='Adam Wentz',
    author_email='adam@adamwentz.com',
    url='https://github.com/awentzonline/keras-rtst/',
    packages=find_packages(),
    install_requires=[
        'Cython>=0.23.4',
        'h5py>=2.5.0',
        'Keras==0.3.3',
        'numpy>=1.10.4',
        'Pillow>=3.1.1',
        'PyYAML>=3.11',
        'scipy>=0.17.0',
        'scikit-learn>=0.17.0',
        'six>=1.10.0',
        'Theano>=0.8.0rc1',
        'keras-vgg-buddy==0.0.5'
    ],
    dependency_links=[
        'https://github.com/Theano/Theano.git@954c3816a40de172c28124017a25387f3bf551b2#egg=Theano',
    ],
    scripts=[
        'scripts/rtst.py',
        'scripts/render-rtst.sh',
        'scripts/rtst-download-training-images.sh',
        'scripts/rtst-gif.sh',
        'scripts/train-rtst.sh',
    ]
)
