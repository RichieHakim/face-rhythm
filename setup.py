from setuptools import find_packages, setup

setup(
    name='face_rhythm',
    packages=find_packages(),
    version='0.1.0',
    description="Project structure for Face Rhythms",
    author='Rich Hakim',
    license='MIT',
    install_requires=['numpy==1.18.3',
                      'torch',
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
                      'h5py==2.10.0',
                      'pynwb',
                      'ipywidgets',
                      'pytest',
                      'Pillow==7.2.0',
                      'SoundFile==0.10.2'
                      ]
)
