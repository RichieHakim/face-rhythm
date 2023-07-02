"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import setuptools
from codecs import open
import os
from pathlib import Path

dir_parent = Path(__file__).parent

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# with open('requirements.txt', 'r') as f:
#     requirements = f.read().splitlines()
# print(requirements)

## Get version number
with open(str(dir_parent / "roicat" / "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().replace("\"", "").replace("\'", "")
            break


setuptools.setup(
    name='face_rhythm',
    version=version,

    description="A pipeline for analysis of facial behavior using optical flow",
    long_description=long_description,
    long_description_content_type='text/markdown',

    # The project's main homepage.
    url='https://github.com/RichieHakim/face-rhythm',

    # Author details
    author='Rich Hakim',

    # Choose your license
    # license='MIT',

    # Supported platforms
    platforms=['Any'],

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        # Pick your license as you wish (should match "license" above)
        # 'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # 'Programming Language :: Python :: 2',
        # 'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.6',
        # 'Programming Language :: Python :: 3.7'
    ],

    # What does your project relate to?
    keywords='face beahvior optical flow',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    # packages=setuptools.find_packages(),
    packages=['face_rhythm'],

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    # install_requires=requirements,

    include_package_data=True,
)