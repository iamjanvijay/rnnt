"""Setup for the rnnt package."""

import setuptools


#with open('README.md') as f:
#    README = f.read()
README = "To use this package please check the instructions in README at - https://github.com/mejanvijay/rnnt"

setuptools.setup(
    author="Janvijay Singh",
    author_email="janvijay.singh.cse14@gmail.com",
    name='rnnt',
    license="MIT",
    description='rnnt is a python package for RNN-Transducer loss in TensorFlow==2.0.',
    version='v0.0.5',
    long_description=README,
    url='https://github.com/mejanvijay/rnnt',
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=['tensorflow==2.6.4', 'numpy==1.18.2'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
)
