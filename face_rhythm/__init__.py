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


## Prepare cv2.imshow
import pkg_resources
installed_packages = {pkg.key for pkg in pkg_resources.working_set}
has_cv2_headless = 'opencv-contrib-python-headless' in installed_packages
has_cv2_normal = 'opencv-contrib-python' in installed_packages
if has_cv2_normal and not has_cv2_headless:
    run_cv2_imshow = True
elif has_cv2_headless and not has_cv2_normal:
    run_cv2_imshow = False
elif has_cv2_headless and has_cv2_normal:
    raise ValueError("Both opencv-contrib-python and opencv-contrib-python-headless are installed. Please uninstall one of them.")
elif not has_cv2_headless and not has_cv2_normal:
    run_cv2_imshow = False
    import warnings
    warnings.warn("Neither opencv-contrib-python nor opencv-contrib-python-headless are installed. Please install one of them to use cv2.imshow().")
else:
    raise ValueError("This should never happen. Please report this error to the developer.")

if run_cv2_imshow:
    def prepare_cv2_imshow():
        """
        This function is necessary because cv2.imshow() 
        can crash the kernel if called after importing 
        av and decord.
        RH 2022
        """
        import numpy as np
        import cv2
        test = np.zeros((1,300,400,3))
        for frame in test:
            cv2.putText(frame, "WELCOME TO FACE RHYTHM!", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, "Prepping CV2", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, "Calling this figure allows cv2.imshow ", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(frame, "to work without crashing if this function", (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(frame, "is called before importing av and decord", (10,190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow('startup', frame)
            cv2.waitKey(1000)
        cv2.destroyWindow('startup')
    prepare_cv2_imshow()


for pkg in __all__:
    exec('from . import ' + pkg)


__version__ = '0.2.3'