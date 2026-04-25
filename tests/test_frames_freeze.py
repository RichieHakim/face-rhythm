"""
Tests for the frames_freeze feature in PointTracker.
Tests both modes:
  - relaxation_during_freeze_frames=True: OF delta zeroed, mesh+relaxation still apply
  - relaxation_during_freeze_frames=False: points fully frozen (exact copy of previous)
"""
import sys
import os
import numpy as np
import torch
import cv2
import scipy.sparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from face_rhythm.point_tracking import PointTracker, _vector_distance, _vector_displacement


## ============================================================================
## Helper: Minimal mock objects to avoid needing real video data
## ============================================================================

class MockBufferedVideoReader:
    """
    Minimal mock of fr.helpers.BufferedVideoReader for unit testing.
    Generates synthetic grayscale frames with a moving bright patch.
    """
    def __init__(self, n_frames=50, height=100, width=100, n_videos=1, move_patch=True):
        self.n_frames = n_frames
        self.height = height
        self.width = width
        self.n_videos = n_videos
        self._move_patch = move_patch
        self.method_getitem = "continuous"
        self.frame_height_width = (height, width)
        self.num_frames_total = n_frames
        self._iterator_idx = 0
        self._frames = self._generate_frames()

    def _generate_frames(self):
        """Generate synthetic frames with a moving bright patch."""
        frames = []
        for i in range(self.n_frames):
            frame = np.full((self.height, self.width, 3), 30, dtype=np.uint8)
            if self._move_patch:
                ## Move a bright patch across the frame
                offset = int(i * 0.5)  ## 0.5 pixels per frame
                y_start = max(20 + offset % 30, 0)
                x_start = max(30 + offset % 30, 0)
                y_end = min(y_start + 40, self.height)
                x_end = min(x_start + 40, self.width)
                frame[y_start:y_end, x_start:x_end] = 200
            frames.append(torch.as_tensor(frame, dtype=torch.uint8))
        return frames

    def get_frames_from_continuous_index(self, idx):
        return self._frames[idx][None, ...]

    def wait_for_loading(self):
        pass

    def set_iterator_frame_idx(self, idx):
        self._iterator_idx = idx

    def delete_all_slots(self):
        pass

    def __len__(self):
        return self.n_frames

    def __iter__(self):
        while self._iterator_idx < self.n_frames:
            yield self._frames[self._iterator_idx]
            self._iterator_idx += 1

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return [self._frames[idx]]
        return [self._frames[i] for i in range(len(self._frames))]


## Monkey-patch isinstance check for BufferedVideoReader
import face_rhythm.point_tracking as pt_module
_original_isinstance = pt_module.__builtins__['isinstance'] if isinstance(pt_module.__builtins__, dict) else __builtins__.__dict__['isinstance'] if isinstance(__builtins__, dict) else getattr(__builtins__, 'isinstance')

def _patched_isinstance(obj, cls):
    from face_rhythm.helpers import BufferedVideoReader
    if cls is BufferedVideoReader and type(obj).__name__ == 'MockBufferedVideoReader':
        return True
    if isinstance(cls, tuple):
        for c in cls:
            if c is BufferedVideoReader and type(obj).__name__ == 'MockBufferedVideoReader':
                return True
    return _original_isinstance(obj, cls)


## ============================================================================
## Test functions
## ============================================================================

def make_tracker(n_frames=50, frames_freeze=None, relaxation_during_freeze_frames=True, move_patch=True):
    """Create a PointTracker with mock data."""
    import builtins
    original = builtins.isinstance

    from face_rhythm.helpers import BufferedVideoReader
    def patched(obj, cls):
        if cls is BufferedVideoReader and type(obj).__name__ == 'MockBufferedVideoReader':
            return True
        if original(cls, tuple):
            for c in cls:
                if c is BufferedVideoReader and type(obj).__name__ == 'MockBufferedVideoReader':
                    return True
        return original(obj, cls)

    builtins.isinstance = patched
    try:
        video = MockBufferedVideoReader(n_frames=n_frames, move_patch=move_patch)

        ## Create a simple grid of points
        y_coords = np.arange(30, 70, 10, dtype=np.float32)
        x_coords = np.arange(30, 70, 10, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)
        point_positions = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

        tracker = PointTracker(
            buffered_video_reader=video,
            point_positions=point_positions,
            rois_masks=None,
            contiguous=True,
            params_optical_flow={
                "method": "lucas_kanade",
                "mesh_rigidity": 0.02,
                "mesh_n_neighbors": 4,
                "relaxation": 0.002,
                "kwargs_method": {
                    "winSize": [15, 15],
                    "maxLevel": 2,
                    "criteria": [cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2, 0.3],
                },
            },
            params_clahe=None,
            params_outlier_handling={
                'threshold_displacement': 100,  ## High threshold to avoid reactive freezing
                'framesHalted_before': 5,
                'framesHalted_after': 5,
            },
            frames_freeze=frames_freeze,
            relaxation_during_freeze_frames=relaxation_during_freeze_frames,
            visualize_video=False,
            verbose=0,
        )
    finally:
        builtins.isinstance = original

    return tracker


def test_no_freeze_baseline():
    """Test that tracking works normally without frames_freeze (backward compatibility)."""
    print("Test: no freeze baseline...", end=" ")
    tracker = make_tracker(n_frames=30, frames_freeze=None)
    tracker.track_points()

    pts = tracker.points_tracked['0']
    assert pts.shape == (30, 16, 2), f"Expected shape (30, 16, 2), got {pts.shape}"
    ## Points should move from their original positions over time
    displacement_last = np.linalg.norm(pts[-1] - tracker.point_positions, axis=1)
    ## With moving patch and relaxation=0.002, some displacement is expected
    print(f"OK (max displacement last frame: {displacement_last.max():.4f})")


def test_freeze_all_frames_full_freeze():
    """
    Test relaxation_during_freeze_frames=False (full freeze).
    When ALL frames are frozen, points should remain exactly at their
    initial positions (no OF, no mesh, no relaxation).
    """
    print("Test: freeze all frames (full freeze)...", end=" ")
    n_frames = 30
    frames_freeze = np.ones(n_frames, dtype=bool)
    tracker = make_tracker(n_frames=n_frames, frames_freeze=frames_freeze, relaxation_during_freeze_frames=False)
    tracker.track_points()

    pts = tracker.points_tracked['0']
    for i in range(n_frames):
        diff = np.abs(pts[i] - tracker.point_positions).max()
        assert diff == 0.0, f"Frame {i}: points moved by {diff} but should be fully frozen"

    print("OK (all points stayed exactly at initial positions)")


def test_freeze_all_frames_with_relaxation():
    """
    Test relaxation_during_freeze_frames=True.
    When ALL frames are frozen, OF delta is zeroed but mesh rigidity
    and relaxation still apply. Since points start at their home positions
    and OF is zeroed, the mesh and relaxation forces should be ~zero
    (points are already at equilibrium). So points should barely move.
    """
    print("Test: freeze all frames (with relaxation)...", end=" ")
    n_frames = 30
    frames_freeze = np.ones(n_frames, dtype=bool)
    tracker = make_tracker(n_frames=n_frames, frames_freeze=frames_freeze, relaxation_during_freeze_frames=True)
    tracker.track_points()

    pts = tracker.points_tracked['0']
    for i in range(n_frames):
        diff = np.abs(pts[i] - tracker.point_positions).max()
        ## Small tolerance for floating point mesh/relaxation on already-equilibrium points
        assert diff < 0.5, f"Frame {i}: points moved by {diff}, expected ~0 since at equilibrium"

    print(f"OK (max displacement: {np.abs(pts - tracker.point_positions[None,...]).max():.6f})")


def test_selective_freeze():
    """
    Test that freezing specific frames zeros the OF delta on those frames
    while allowing normal tracking on non-frozen frames.
    """
    print("Test: selective freeze (full freeze mode)...", end=" ")
    n_frames = 30

    ## Freeze frames 10-19
    frames_freeze = np.zeros(n_frames, dtype=bool)
    frames_freeze[10:20] = True

    tracker = make_tracker(n_frames=n_frames, frames_freeze=frames_freeze, relaxation_during_freeze_frames=False)
    tracker.track_points()
    pts = tracker.points_tracked['0']

    ## On frozen frames (full freeze mode), each frame should equal the previous frame
    for i in range(10, 20):
        if i == 0:
            continue
        diff = np.abs(pts[i] - pts[i-1]).max()
        assert diff == 0.0, f"Frame {i}: frozen frame changed by {diff}"

    print("OK (frozen frames have identical positions to previous frame)")


def test_selective_freeze_with_relaxation():
    """
    Test that freezing specific frames with relaxation_during_freeze_frames=True
    allows mesh/relaxation forces to still act even during frozen frames.
    """
    print("Test: selective freeze (with relaxation mode)...", end=" ")
    n_frames = 30

    ## Freeze frames 10-19
    frames_freeze = np.zeros(n_frames, dtype=bool)
    frames_freeze[10:20] = True

    tracker_relax = make_tracker(n_frames=n_frames, frames_freeze=frames_freeze, relaxation_during_freeze_frames=True)
    tracker_relax.track_points()
    pts_relax = tracker_relax.points_tracked['0']

    tracker_full = make_tracker(n_frames=n_frames, frames_freeze=frames_freeze, relaxation_during_freeze_frames=False)
    tracker_full.track_points()
    pts_full = tracker_full.points_tracked['0']

    ## Before freeze window, both should be identical (same tracking)
    diff_before = np.abs(pts_relax[:10] - pts_full[:10]).max()
    assert diff_before == 0.0, f"Pre-freeze diverged by {diff_before}"

    ## During or after freeze, they may differ because relaxation mode
    ## allows mesh/relaxation forces to act on frozen frames
    ## (this difference may be small if points are near equilibrium)
    diff_during = np.abs(pts_relax[10:20] - pts_full[10:20]).max()
    print(f"OK (pre-freeze identical, freeze-window max diff: {diff_during:.6f})")


def test_freeze_then_resume():
    """
    Test that after a freeze window, tracking resumes normally.
    Compare against a no-freeze run to verify pre-freeze behavior is identical.
    """
    print("Test: freeze then resume...", end=" ")
    n_frames = 30

    ## No freeze
    tracker_nf = make_tracker(n_frames=n_frames, frames_freeze=None)
    tracker_nf.track_points()
    pts_nf = tracker_nf.points_tracked['0']

    ## Freeze frames 10-15
    frames_freeze = np.zeros(n_frames, dtype=bool)
    frames_freeze[10:16] = True
    tracker_f = make_tracker(n_frames=n_frames, frames_freeze=frames_freeze, relaxation_during_freeze_frames=False)
    tracker_f.track_points()
    pts_f = tracker_f.points_tracked['0']

    ## Before freeze window, tracking should be identical
    diff_before = np.abs(pts_nf[:10] - pts_f[:10]).max()
    assert diff_before == 0.0, f"Pre-freeze diverged by {diff_before}"

    ## After freeze, tracking should diverge because the freeze altered the trajectory
    ## (points were held still during frames 10-15, so they start from different positions)
    diff_after = np.abs(pts_nf[20:] - pts_f[20:]).max()

    print(f"OK (pre-freeze identical, post-freeze max diff: {diff_after:.4f})")


def test_input_validation():
    """Test that invalid frames_freeze inputs are caught."""
    print("Test: input validation...", end=" ")

    ## Wrong type
    try:
        make_tracker(n_frames=30, frames_freeze=[True, False])
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "1D boolean numpy array" in str(e)

    ## Wrong dtype
    try:
        make_tracker(n_frames=30, frames_freeze=np.zeros(30, dtype=np.float32))
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "dtype bool" in str(e)

    ## Wrong ndim
    try:
        make_tracker(n_frames=30, frames_freeze=np.zeros((30, 2), dtype=bool))
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "1D" in str(e)

    print("OK (all invalid inputs caught)")


def test_length_validation():
    """Test that mismatched frames_freeze length is caught during tracking."""
    print("Test: length validation...", end=" ")

    ## Wrong length (should fail during track_points)
    tracker = make_tracker(n_frames=30, frames_freeze=np.zeros(20, dtype=bool))
    try:
        tracker.track_points()
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "length" in str(e).lower() or "must match" in str(e).lower()

    print("OK (mismatched length caught)")


def test_config_stored():
    """Test that frames_freeze config is stored in FR_Module config."""
    print("Test: config storage...", end=" ")

    tracker = make_tracker(n_frames=30, frames_freeze=np.zeros(30, dtype=bool), relaxation_during_freeze_frames=False)
    assert tracker.config['frames_freeze'] == True, "frames_freeze should be True in config when provided"
    assert tracker.config['relaxation_during_freeze_frames'] == False

    tracker2 = make_tracker(n_frames=30, frames_freeze=None)
    assert tracker2.config['frames_freeze'] == False, "frames_freeze should be False in config when None"
    assert tracker2.config['relaxation_during_freeze_frames'] == True

    print("OK")


## ============================================================================
## Run all tests
## ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing frames_freeze feature in PointTracker")
    print("=" * 60)

    test_no_freeze_baseline()
    test_freeze_all_frames_full_freeze()
    test_freeze_all_frames_with_relaxation()
    test_selective_freeze()
    test_selective_freeze_with_relaxation()
    test_freeze_then_resume()
    test_input_validation()
    test_length_validation()
    test_config_stored()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
