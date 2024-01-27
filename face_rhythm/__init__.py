## Import packages
__all__=[
    # 'analysis',
    # 'comparisons',
    'decomposition',
    'h5_handling',
    'helpers',
    # 'neural',
    # 'optic_flow',
    'pipelines',
    'point_tracking',
    'project',
    'rois',
    'spectral_analysis',
    'util',
    'visualization',
    'data_importing',
    # 'tests',
]

import torch  ## For some reason, it crashes if I don't import torch before other packages... RH 20221128

for pkg in __all__:
    exec('from . import ' + pkg)

__version__ = '0.1.4'