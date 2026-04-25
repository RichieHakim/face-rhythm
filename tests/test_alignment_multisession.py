"""Smoke tests for face_rhythm.alignment_multisession.

These tests validate three invariants that the port must hold:

1. The module imports cleanly even when the optional ``romatch`` dependency
   is not installed.
2. ``Aligner(...)`` can be instantiated with the notebook's canonical kwargs.
3. ``Aligner.fit_geometric(method='ECC_cv2', ...)`` runs end-to-end on two
   tiny synthetic images (the OpenCV-only path, no model downloads).
4. ``Aligner.fit_geometric(method='RoMa', ...)`` raises a clean
   ``ImportError`` if the ``multisession`` extra isn't installed, and is
   otherwise skipped (we don't exercise the RoMa model in CI since it
   downloads ~1.5 GB of weights).
"""

import sys

import numpy as np
import pytest


def test_import_module_without_romatch(monkeypatch):
    """``alignment_multisession`` must import cleanly without ``romatch`` present."""
    ## Force ``import romatch`` to raise within this test.
    monkeypatch.setitem(sys.modules, 'romatch', None)

    ## Import a fresh copy so any conditional top-level logic re-runs.
    sys.modules.pop('face_rhythm.alignment_multisession', None)
    from face_rhythm import alignment_multisession  ## noqa: F401

    ## Smoke: key symbols are present.
    assert hasattr(alignment_multisession, 'Aligner')
    assert hasattr(alignment_multisession, 'ECC_cv2')
    assert hasattr(alignment_multisession, 'PhaseCorrelationRegistration')
    assert hasattr(alignment_multisession, 'NullRegistration')
    assert hasattr(alignment_multisession, 'RoMa')
    assert hasattr(alignment_multisession, 'ImageAlignmentChecker')


def test_aligner_can_be_instantiated():
    """The notebook's canonical constructor should work without errors."""
    from face_rhythm.alignment_multisession import Aligner

    aligner = Aligner(
        use_match_search=True,
        all_to_all=True,
        radius_in=15,
        radius_out=75,
        order=5,
        z_threshold=50,
        device='cpu',
        verbose=0,
    )
    assert aligner.use_match_search is True
    assert aligner.all_to_all is True
    assert aligner.radius_in == 15
    assert aligner.radius_out == 75
    assert aligner.z_threshold == 50
    assert aligner.remappingIdx_geo is None
    assert aligner.device == 'cpu'


def test_fit_geometric_ecc_cv2():
    """End-to-end ECC_cv2 alignment on synthetic translated images."""
    from face_rhythm.alignment_multisession import Aligner

    ## Build a 32x32 image with structured content (mixture of rectangles) so
    ## the ECC solver has actual gradients to match.
    H, W = 32, 32
    rng = np.random.default_rng(42)
    im_template = rng.uniform(low=0.0, high=0.2, size=(H, W)).astype(np.float32)
    im_template[8:20, 10:22] = 0.9
    im_template[4:6, 12:18] = 0.5

    ## im_moving = im_template translated by (dx, dy) = (2, 1). We expect ECC
    ## to recover a warp that approximately undoes this shift.
    shift_x, shift_y = 2, 1
    im_moving = np.zeros_like(im_template)
    im_moving[shift_y:, shift_x:] = im_template[:H - shift_y, :W - shift_x]

    aligner = Aligner(
        use_match_search=False,
        all_to_all=False,
        radius_in=3,
        radius_out=8,
        order=3,
        z_threshold=0.0,  ## accept any alignment — don't trigger match-search
        device='cpu',
        verbose=0,
    )
    remap = aligner.fit_geometric(
        template=0,
        ims_moving=[im_template, im_moving],
        template_method='image',
        mask_borders=(0, 0, 0, 0),
        method='ECC_cv2',
        kwargs_method={'ECC_cv2': {
            'mode_transform': 'translation',
            'n_iter': 500,
            'termination_eps': 1e-8,
            'gaussFiltSize': 1,
            'auto_fix_gaussFilt_step': 10,
        }},
        constraint='affine',
        kwargs_RANSAC={'inl_thresh': 2.0, 'max_iter': 10, 'confidence': 0.99},
        verbose=False,
    )
    ## Sanity: correct shape + shape of each entry.
    assert isinstance(remap, list)
    assert len(remap) == 2
    for r in remap:
        assert r.shape == (H, W, 2)
        assert r.dtype == np.float32 or np.issubdtype(r.dtype, np.floating)
    ## aligner.remappingIdx_geo should be populated.
    assert aligner.remappingIdx_geo is not None
    assert len(aligner.remappingIdx_geo) == 2
    ## transform_images returns a list of length 2.
    ims_out = aligner.transform_images(
        ims_moving=[im_template, im_moving],
        remappingIdx=aligner.remappingIdx_geo,
    )
    assert isinstance(ims_out, list) and len(ims_out) == 2
    assert all(im.shape == (H, W) for im in ims_out)


def test_roma_backend_raises_without_romatch(monkeypatch):
    """Without ``romatch`` installed, the RoMa backend must raise a clear ImportError."""
    monkeypatch.setitem(sys.modules, 'romatch', None)
    sys.modules.pop('face_rhythm.alignment_multisession', None)
    from face_rhythm.alignment_multisession import Aligner

    aligner = Aligner(
        use_match_search=False, all_to_all=False,
        radius_in=3, radius_out=8, order=3, z_threshold=0.0,
        device='cpu', verbose=0,
    )
    im = np.zeros((16, 16), dtype=np.float32)
    im[4:12, 4:12] = 1.0
    with pytest.raises(ImportError, match='multisession'):
        aligner.fit_geometric(
            template=0,
            ims_moving=[im, im.copy()],
            template_method='image',
            method='RoMa',
            kwargs_method={'RoMa': {'model_type': 'outdoor', 'n_points': 100, 'batch_size': 50}},
            constraint='affine',
            verbose=False,
        )


def test_roma_backend_is_reachable_when_romatch_is_installed():
    """With ``romatch`` available, the RoMa entry point is reachable.

    We don't actually run RoMa in CI (it downloads ~1.5 GB of weights). We
    simply confirm that the ImportError guard does NOT fire — any exception
    from the constructor after the import succeeded is acceptable here.
    """
    pytest.importorskip('romatch')
    from face_rhythm.alignment_multisession import RoMa

    ## We call RoMa with a fake URL that will fail the hash check rather than
    ## download. This exercises the import path but stops before downloading.
    bogus = {
        "romatch": {"outdoor": {"url": "file:///dev/null", "hash": "0", "filename": "never.pth"}},
        "dinov2": {"url": "file:///dev/null", "hash": "0", "filename": "never.pth"},
    }
    ## Any exception other than ImportError(multisession) is fine — we just
    ## want to prove the extra is installed and the import path runs.
    with pytest.raises(Exception) as excinfo:
        RoMa(
            model_type='outdoor',
            n_points=100,
            batch_size=50,
            device='cpu',
            weight_urls=bogus,
            fallback_weight_urls=bogus,
            verbose=False,
        )
    assert 'multisession' not in str(excinfo.value), (
        f"Expected past the ImportError guard, but got: {excinfo.value}"
    )
