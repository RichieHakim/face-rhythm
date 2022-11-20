## Import packages
__all__=[
    'analysis',
    'comparisons',
    'neural',
    'optic_flow',
    'project',
    'tests',
    'utils',
    'visualize',
    'data_importing',
]

for pkg in __all__:
    exec('from . import ' + pkg)

__version__ = '0.1.0'