#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import re

import setuptools
from pathlib import Path

root_directory = Path(__file__).parent
readme_filepath = root_directory.joinpath('README.md')
long_description = readme_filepath.read_text()


def get_version():
    version_filepath = Path('src', 'highdicom', 'version.py')
    with io.open(version_filepath, 'rt', encoding='utf8') as f:
        version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)
    return version


setuptools.setup(
    name='highdicom',
    version=get_version(),
    description='High-level DICOM abstractions.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Markus D. Herrmann',
    maintainer='Markus D. Herrmann',
    url='https://github.com/herrmannlab/highdicom',
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Development Status :: 4 - Beta',
    ],
    include_package_data=True,
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    python_requires='>=3.6',
    install_requires=[
        'pydicom>=2.3.0',
        'numpy>=1.19',
        'pillow>=8.3',
        'pillow-jpls>=1.0',
        'pylibjpeg>=1.4',
        'pylibjpeg-libjpeg>=1.3',
        'pylibjpeg-openjpeg>=1.2',
    ],
)
