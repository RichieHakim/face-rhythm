"""
Regression tests for ``face_rhythm.point_tracking.PointTracker``.

Locks the fix for GitHub issue #80: when ``rois_masks=None`` and the
``BufferedVideoReader`` is in ``method_getitem='by_video'`` mode (the mode the
standard demo/pipeline uses), ``PointTracker`` must build a full ``(H, W)`` ROI
mask. Previously it indexed ``buffered_video_reader[0][0]`` and sliced
``.shape[:2]``; in 'by_video' mode that index returns a 4D batch
``(1, H, W, C)`` rather than a 3D frame ``(H, W, C)``, producing a degenerate
``(1, H)`` mask that crashes the optical-flow masking with a dimension
mismatch. The fix uses the reader's mode/backend-independent
``frame_height_width`` metadata instead.
"""
from pathlib import Path

import numpy as np

from face_rhythm import helpers, point_tracking


## Use 'decord' rather than 'torchcodec': the integration test treats decord as a
## hard dependency (no skipif), so it is always available in CI, whereas
## torchcodec is optional. The bug is backend-independent (decord and torchcodec
## return identical frame shapes); the trigger is method_getitem, not the backend.
BACKEND = "decord"


def _make_reader(dir_data_test, method_getitem):
    """
    Builds a ``BufferedVideoReader`` over a single demo input video.

    Args:
        dir_data_test (str):
            Directory containing the test data (``inputs/`` with the demo
            ``.avi`` videos). Provided by the ``dir_data_test`` fixture.
        method_getitem (str):
            Indexing mode passed to ``BufferedVideoReader``; either
            ``'continuous'`` or ``'by_video'``.

    Returns:
        (helpers.BufferedVideoReader):
            Reader over the first demo video.
    """
    path_video = str(Path(dir_data_test).resolve() / "inputs" / "demo_mouse0322N20230430cam4_1.avi")
    return helpers.BufferedVideoReader(
        paths_videos=[path_video],
        buffer_size=1000,
        method_getitem=method_getitem,
        backend=BACKEND,
        verbose=0,
    )


def test_pointTracker_defaultMask_shape_byVideo(dir_data_test):
    """
    Regression for issue #80: with ``rois_masks=None`` in 'by_video' mode, the
    default mask must equal the reader's ``(H, W)`` metadata (not a degenerate
    ``(1, H)``). This is the precise, mode-independent invariant.

    The ``method_getitem`` is set to 'by_video' *before* constructing the
    tracker because the mask is built in ``PointTracker.__init__`` from the
    reader's mode as passed in.
    """
    reader = _make_reader(dir_data_test, method_getitem="by_video")
    H, W = reader.frame_height_width
    ## Sanity-check the trigger condition: in 'by_video' mode the index returns a
    ## 4D batch, and the video is non-square so a degenerate (1, H) mask would
    ## differ from the correct (H, W) mask.
    assert np.asarray(reader[0][0]).ndim == 4, "Expected a 4D batch from reader[0][0] in 'by_video' mode"
    assert H != W, "This regression relies on a non-square test video to expose the degenerate mask"

    ## A few seed points inside the frame (>= mesh_n_neighbors=8 to keep mesh setup happy).
    point_positions = (np.array([[c * W, r * H] for r in (0.3, 0.5, 0.7) for c in (0.3, 0.5, 0.7)], dtype=np.float32))

    pt = point_tracking.PointTracker(
        buffered_video_reader=reader,
        point_positions=point_positions,
        rois_masks=None,
        contiguous=False,
        visualize_video=False,
        verbose=0,
    )

    assert tuple(pt.mask.shape) == tuple(reader.frame_height_width), (
        f"Default mask shape {tuple(pt.mask.shape)} != reader.frame_height_width "
        f"{tuple(reader.frame_height_width)} (issue #80 degenerate-mask regression)."
    )

    ## Smoke check: exercise the optical-flow formatter/masking path that raised
    ## issue #80's Error #1 (``vid * mask[None, :, :]``) on the first frame. With
    ## the degenerate mask this raised a dim mismatch; with the fix it returns a
    ## (batch, H, W) array. (One frame only — avoids running the full video.)
    vid_first = pt.buffered_video_reader.get_frames_from_continuous_index(0)
    vid_formatted = pt._format_decordTorchVideo_for_opticalFlow(vid=vid_first, mask=pt.mask)
    assert vid_formatted.shape[-2:] == (H, W), (
        f"Formatted frame spatial dims {vid_formatted.shape[-2:]} != (H, W) ({(H, W)})."
    )


def test_pointTracker_defaultMask_shape_continuous(dir_data_test):
    """
    The previously-correct 'continuous' path must keep producing an ``(H, W)``
    mask (the fix is behavior-preserving there).
    """
    reader = _make_reader(dir_data_test, method_getitem="continuous")
    H, W = reader.frame_height_width
    point_positions = np.array([[0.5 * W, 0.5 * H]], dtype=np.float32)

    pt = point_tracking.PointTracker(
        buffered_video_reader=reader,
        point_positions=point_positions,
        rois_masks=None,
        contiguous=False,
        visualize_video=False,
        verbose=0,
    )

    assert tuple(pt.mask.shape) == tuple(reader.frame_height_width)
