## setup.py file for roicat
from pathlib import Path
import copy
import platform

from distutils.core import setup

## Get the parent directory of this file
dir_parent = Path(__file__).parent

## Get requirements from requirements.txt
def read_requirements():
    with open(str(dir_parent / "requirements.txt"), "r") as req:
        content = req.read()  ## read the file
        requirements = content.split("\n") ## make a list of requirements split by (\n) which is the new line character

    ## Filter out any empty strings from the list
    requirements = [req for req in requirements if req]
    ## Filter out any lines starting with #
    requirements = [req for req in requirements if not req.startswith("#")]
    ## Remove any commas, quotation marks, and spaces from each requirement
    requirements = [req.replace(",", "").replace("\"", "").replace("\'", "").strip() for req in requirements]

    return requirements
deps_all = read_requirements()


## Dependencies: latest versions of requirements
### remove everything starting and after the first =,>,<,! sign
deps_names = [req.split('=')[0].split('>')[0].split('<')[0].split('!')[0] for req in deps_all]
deps_all_dict = dict(zip(deps_names, deps_all))
deps_all_latest = dict(zip(deps_names, deps_names))


## Operating system specific dependencies
### OpenCV >= 4.9 is not supported on macOS < 12
system, version = platform.system(), platform.mac_ver()[0]
if system == "Darwin" and version and ('opencv_contrib_python' in deps_all_dict):
    if tuple(map(int, version.split('.'))) < (12, 0, 0):
        version_opencv_macosSub12 = "opencv_contrib_python<=4.8.1.78"
        deps_all_dict['opencv_contrib_python'], deps_all_latest['opencv_contrib_python'] = version_opencv_macosSub12, version_opencv_macosSub12


## Make different versions of dependencies
### Also pull out the version number from the requirements (specified in deps_all_dict values).
deps_core = [deps_all_dict[dep] for dep in [
    'numpy',
    'jupyter',
    'notebook',
    'tensorly',
    'opencv-contrib-python',
    'matplotlib',
    'scikit-learn',
    'scikit-image',
    'pyyaml',
    'tqdm',
    'h5py',
    'ipywidgets',
    'Pillow',
    'eva-decord',
    'natsort',
    'pandas',
    'tables',
    'einops',
    'pytest',
    'torch',
    'torchvision',
    'torchaudio',
    'nvidia-ml-py3',
    'py-cpuinfo',
    'GPUtil',
    'psutil',
]]


## Make versions with cv2 headless (for servers)
deps_all_dict_cv2Headless = copy.deepcopy(deps_all_dict)
deps_all_dict_cv2Headless['opencv-contrib-python'] = 'opencv-contrib-python-headless' + deps_all_dict_cv2Headless['opencv-contrib-python'][21:]
deps_all_latest_cv2Headless = copy.deepcopy(deps_all_latest)
deps_all_latest_cv2Headless['opencv-contrib-python'] = 'opencv-contrib-python-headless'


## Get README.md
with open(str(dir_parent / "README.md"), "r") as f:
    readme = f.read()

## Get face-rhythm version number
with open(str(dir_parent / "face_rhythm" / "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().replace("\"", "").replace("\'", "")
            break


setup(
    name='face_rhythm',
    version=version,
    keywords=['neuroscience', 'neuroimaging', 'machine learning'],

    description="A pipeline for analysis of facial behavior using optical flow",
    long_description=readme,
    long_description_content_type='text/markdown',

    # The project's main homepage.
    url='https://github.com/RichieHakim/face-rhythm',

    # Author details
    author='Rich Hakim',

    # Choose your license
    license='LICENSE',

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

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    # packages=setuptools.find_packages(),
    packages=['face_rhythm'],

    install_requires=[],

    extras_require={
        'all': list(deps_all_dict.values()),
        'all_latest': list(deps_all_latest.values()),
        'all_cv2Headless': list(deps_all_dict_cv2Headless.values()),
        'all_latest_cv2Headless': list(deps_all_latest_cv2Headless.values()),
        'core': deps_core,
    },

    include_package_data=True,
)