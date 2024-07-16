
from setuptools import setup, find_packages

setup(
    name='IPFInitFit',
    version='0.6',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for Iterative Proportional Fitting (IPF) with optimal initial weights.',
    url='https://github.com/yourusername/IPFInitFit',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
