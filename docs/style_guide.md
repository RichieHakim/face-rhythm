---
orphan: true
---

# face-rhythm Docstring Style Guide

This guide defines the docstring contract for the `face_rhythm` package. It is
designed to be pasted as a system prompt to an LLM that rewrites docstrings
across the codebase, one function or class at a time. The primary goal is
**consistency** across all modules so that the auto-generated Sphinx API page
renders cleanly under Google-style + Napoleon. The LLM returns the corrected
definition and docstring **only** — the function body is echoed unchanged or
omitted, and function/class/argument names must never be renamed.

---

## Function signature formatting

1. Each parameter goes on its own line if there is more than one parameter.
2. Include a default value in the signature only when the default is not
   obvious from context.
3. Use type hints for every parameter and the return value. Prefer the
   `typing` module style (`Optional[...]`, `Union[...]`, `List[...]`,
   `Dict[...]`, `Tuple[...]`, `Callable[...]`).
4. Put a trailing comma after every argument, including the last one.
5. If the type hint is exotic (e.g. `matplotlib.colors.LinearSegmentedColormap`,
   `decord.VideoReader`, `torchcodec.decoders.VideoDecoder`), fall back to
   `object`. `torch.Tensor`, `numpy.ndarray`, and `scipy.sparse` matrices are
   **not** exotic and should be used directly.

---

## Docstring rules

1. The function/method description must be concise and clear (one or two
   sentences). Preserve any author tag that already exists at the end of the
   headline sentence (e.g. `RH 2023`); do not add new ones.
2. Use Google-style docstrings (Sphinx + Napoleon compatible). Start the
   description of every argument and return value on a new indented line under
   its name+type entry.
3. Use the following modifiers consistently:
   - **Bold** for important words: `**important**`
   - *Italics* for array shapes, dtypes, and other special values:
     `shape: *(n_frames, H, W)*`, `dtype: *float32*`
   - `Code` formatting for code, literal arguments, and default values:
     ``Default is ``None`` ``, ``If ``True`` then ...``
4. When enumerating choices for a parameter, use Sphinx-style bullet points
   prefixed with `* `, with a `\n` line break before the first bullet and after
   the last (see the `backend` argument in the worked example below).
5. Use proper grammar, punctuation, spelling, and capitalization. Avoid filler
   words and unnecessary line breaks.
6. Class docstrings should describe `__init__` (move `__init__` to the top of
   the class body if it is not already there) and list any class attributes
   in an `Attributes:` block placed **after** Args, Returns, and any Example.
7. If a function returns `None`, omit the `Returns` section entirely. If it
   returns a value but does not explicitly bind it to a name, invent a short,
   descriptive variable name for the docstring.
8. Append the default value to the end of each argument's description in
   parentheses, even if it is also visible in the signature, e.g.
   `(Default is ``1000``)`.
9. Single return uses this shape:
    ```
    Returns:
        (np.ndarray):
            frames (np.ndarray):
                Stacked frames. shape: *(N, H, W, C)*, dtype: *uint8*.
    ```
10. Multiple returns use this shape:
    ```
    Returns:
        (tuple): tuple containing:
            points (np.ndarray):
                Tracked point coordinates. shape: *(n_frames, n_points, 2)*.
            status (np.ndarray):
                Validity flag per point per frame. shape: *(n_frames, n_points)*,
                dtype: *bool*.
    ```
11. **Do not change** any function, class, or argument names. Renaming breaks
    dependent code (notebooks, pipelines, downstream callers).
12. Demo / example code is only required when the call is non-trivial (a class
    used through several methods, or a function with several returns). Place
    the `Example:` block after `Args` and `Returns` and format it as:
    ```
    Example:
        .. highlight:: python
        .. code-block:: python

            reader = BufferedVideoReader(paths_videos=paths, buffer_size=1000)
            frames = reader.get_frames(idx_video=0, idx_frames=slice(0, 100))
    ```
13. Do not indent continuation lines of a docstring paragraph; align them with
    the first line of the same paragraph.

---

## Worked examples

### Class with one method

```python
class BufferedVideoReader:
    """
    Reads frames from one or more videos with an in-memory frame buffer and
    optional background prefetching. RH 2023

    Args:
        paths_videos (List[str]):
            Paths to the video files, in the order they should be concatenated.
        buffer_size (int):
            Number of frames held in memory per buffer slot. (Default is ``1000``)
        backend (str):
            Decoder backend to use. Either \n
            * ``'decord'``: Bundled decord backend.
            * ``'torchcodec'``: torchcodec backend with the issue #905
              workaround. \n
            (Default is ``'decord'``)

    Attributes:
        num_frames_total (int):
            Total number of frames across all videos.
        metadata (dict):
            Per-video metadata: shape, fps, dtype.
    """
    def __init__(
        self,
        paths_videos: List[str],
        buffer_size: int = 1000,
        backend: str = 'decord',
    ):
        """Initializes the reader, opens each video, and starts the prefetcher."""

    def get_frames(
        self,
        idx_video: int,
        idx_frames: Union[int, slice, List[int], np.ndarray],
    ) -> np.ndarray:
        """
        Returns frames from a single video by index, slice, or index array.

        Args:
            idx_video (int):
                Index of the video within ``paths_videos``.
            idx_frames (Union[int, slice, List[int], np.ndarray]):
                Frame indices to return, scoped to the chosen video.

        Returns:
            (np.ndarray):
                frames (np.ndarray):
                    Stacked frames. shape: *(N, H, W, C)*, dtype: *uint8*.
        """
```

### Standalone function

```python
def find_paths(
    directory: str,
    filename_strMatch: Optional[str] = None,
    depth: int = 1,
) -> List[str]:
    """
    Recursively finds files in ``directory`` whose names match a regex.

    Args:
        directory (str):
            Root directory to search.
        filename_strMatch (Optional[str]):
            Regex that filenames must match. If ``None``, all files are
            returned. (Default is ``None``)
        depth (int):
            Maximum subdirectory depth to descend. (Default is ``1``)

    Returns:
        (List[str]):
            paths (List[str]):
                Absolute paths to the matched files.
    """
```

---

## face-rhythm scope

The package contains these top-level modules: `alignment`,
`alignment_multisession`, `data_importing`, `decomposition`, `h5_handling`,
`helpers`, `pipelines`, `point_tracking`, `project`, `rois`,
`spectral_analysis`, `util`, `visualization`.

Hint these recurring types directly (not as `object`): `numpy.ndarray`,
`torch.Tensor`, `pathlib.Path`, `scipy.sparse` matrices, `h5py.File`,
`matplotlib.figure.Figure`, `matplotlib.axes.Axes`, `pandas.DataFrame`.
Decoder objects from `decord` and `torchcodec` are exotic — hint them as
`object`.
