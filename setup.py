#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import re

import setuptools

with io.open('src/highdicom/version.py', 'rt', encoding='utf8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)


setuptools.setup(
    name='highdicom',
    version=version,
    description='High-level abstractions for the DICOM standard.',
    author='Markus D. Herrmann',
    maintainer='Markus D. Herrmann',
    url='https://github.com/mghcomputationalpathology/highdicom',
    license='MIT',
    platforms=['Linux', 'MacOS', 'Windows'],
    classifiers=[
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Development Status :: 3 - Alpha',
    ],
    include_package_data=True,
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    setup_requires=[
        'pytest-runner>=3.0',
    ],
    extras_require={
        'docs': [
            'sphinx>=1.7.1',
            'sphinx-pyreverse>=0.0.12',
            'sphinxcontrib-autoprogram>=0.1.4',
            'sphinx_rtd_theme>=0.2.4'
        ]
    },
    tests_require=[
        'pytest>=3.3',
        'pytest-flake8>=0.9',
        'tox>=2.9'
    ],
    install_requires=[
        'pydicom>=1.4.1',
        'numpy>=1.0',
        'pillow>=6.0'
    ],
)
