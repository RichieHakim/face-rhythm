from setuptools import find_packages, setup

setup(
    name='face_rhythm',
    packages=find_packages(),
    version='0.1.0',
    description="Project structure for Face Rhythms",
    author='Rich Hakim',
    license='MIT',
    install_requires=['torch',
                      'torchvision',
                      'torchaudio',
                      'jupyterlab',
                      'tensorly',
                      'opencv-python',
                      'imageio==2.9.0',
                      'matplotlib',
                      'scikit-learn',
                      'scikit-image',
                      'librosa',
                      'pyyaml',
                      'imageio-ffmpeg',
                      'tqdm',
                      'h5py',
                      'pynwb',
                      'ipywidgets',
                      'pytest'
                      ]
)
