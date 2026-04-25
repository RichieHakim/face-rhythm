"""ROI selection, point-grid generation, and image warping / registration.

Provides the ``ROIs`` class (choose face regions via GUI, file, or explicit
dict), the ``ImageAlignmentChecker`` and registration helpers used by the
alignment module, and the interactive ``_Select_ROI`` Plotly/ipywidgets GUI.
"""

from pathlib import Path
import warnings
import functools

import cv2
import numpy as np
import scipy.interpolate
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from .util import FR_Module
from . import h5_handling, helpers


class ROIs(FR_Module):
    """
    Container for one or more face ROIs and the tracking points sampled within
    them. Supports three construction modes: interactive GUI, loading a saved
    ``ROIs.h5`` file, or building from explicit polygon coordinates. RH 2022

    Args:
        select_mode (str):
            How to populate the ROIs. One of \n
            * ``'gui'``: Launch the interactive Plotly/ipywidgets selector.
              ``exampleImage`` must be provided.
            * ``'file'``: Load ``mask_images``, ``roi_points``, and
              ``exampleImage`` from a previously saved ``ROIs.h5`` file.
              ``path_file`` must be provided.
            * ``'custom'``: Build masks from explicit polygon coordinates.
              ``coords_rois`` and ``exampleImage`` must be provided. \n
            (Default is ``'gui'``)
        exampleImage (np.ndarray):
            Image to display in the GUI or to define the canvas size for
            ``'custom'`` mode. Only used when ``select_mode`` is ``'gui'`` or
            ``'custom'``. (Default is ``None``)
        path_file (str):
            Path to a saved ``ROIs.h5`` file. Only used when ``select_mode``
            is ``'file'``. (Default is ``None``)
        coords_rois (dict):
            Dictionary mapping ROI names (e.g. ``'ROI_0'``, ``'ROI_1'``) to
            polygon vertices, given as either an ``np.ndarray`` of shape
            *(N, 2)* or a list of ``[x, y]`` pairs. Only used when
            ``select_mode`` is ``'custom'``. (Default is ``None``)
        point_positions (np.ndarray):
            Optional pre-computed array of tracking point positions, shape
            *(n_points, 2)*. (Default is ``None``)
        mask_images (dict):
            Dictionary mapping mask names to 2D boolean ``np.ndarray`` masks
            with the same height and width as the videos. (Default is
            ``None``)
        verbose (int):
            Verbosity level. One of \n
            * ``0``: No output.
            * ``1``: Warnings only.
            * ``2``: All output. \n
            (Default is ``1``)

    Attributes:
        exampleImage (np.ndarray):
            The reference image associated with the ROIs.
        roi_points (dict):
            Polygon vertices for each ROI keyed by name.
        mask_images (dict):
            Boolean masks for each ROI keyed by name.
        point_positions (np.ndarray):
            Tracking point positions, shape *(n_points, 2)*.
        num_points (int):
            Total number of tracking points.
        img_hw (tuple):
            Height and width of ``exampleImage``.
    """
    def __init__(
        self,
        select_mode="gui",

        exampleImage=None,
        path_file=None,
        coords_rois=None,
        point_positions=None,
        mask_images=None,
        verbose=1,
    ):
        """Initializes the ROIs container according to ``select_mode``."""
        super().__init__()
        self._select_mode = select_mode
        self.exampleImage = exampleImage
        self._path_file = path_file
        self.roi_points = coords_rois
        self.mask_images = mask_images
        self._verbose = int(verbose)
        self.point_positions = None

        self.img_hw = self.exampleImage.shape[:2] if self.exampleImage is not None else None

        ## Assert that the correct arguments are provided for the select_mode
        assert isinstance(select_mode, str), "FR ERROR: select_mode must be a string."
        if (select_mode == "gui"):
            assert exampleImage is not None, "FR ERROR: 'exampleImage' must be provided for select_mode 'gui'."
            assert path_file is None, "FR ERROR: 'path_file' must not be provided for select_mode 'gui'."
        elif select_mode == "file":
            assert self._path_file is not None, "FR ERROR: 'path_file' must be provided for select_mode 'file'."
            assert isinstance(self._path_file, str), "FR ERROR: 'path_file' must be a string."
            assert Path(self._path_file).exists(), f"FR ERROR: 'path_file' does not exist: {self._path_file}"
        elif select_mode == "custom":
            assert self.roi_points is not None, "FR ERROR: 'points' must be provided for select_mode 'custom'."
            assert self.exampleImage is not None, "FR ERROR: 'exampleImage' must be provided for select_mode 'custom'."
            assert isinstance(self.roi_points, dict), "FR ERROR: 'points' must be a dictionary."
            assert all([isinstance(v, (np.ndarray, list)) for v in self.roi_points.values()]), "FR ERROR: 'points' must be a dictionary of numpy arrays or lists."
            if isinstance(self.roi_points[list(self.roi_points.keys())[0]], np.ndarray):
                assert all([v.shape[1] == 2 for v in self.roi_points.values()]), "FR ERROR: 'points' must be a dictionary of numpy arrays of shape (N, 2)."
            elif isinstance(self.roi_points[list(self.roi_points.keys())[0]], list):
                assert all([len(v) == 2 for v in self.roi_points.values()]), "FR ERROR: 'points' must be a dictionary of lists of length 2."
        else:
            raise ValueError("FR ERROR: 'select_mode' must be one of 'gui', 'file', 'custom'.")


        if select_mode == "gui":
            print(f"FR: Initializing GUI...") if self._verbose > 1 else None
            self._gui = _Select_ROI(exampleImage)
            self.roi_points = self._gui.selected_points
            self.mask_images = self._gui.mask_frames
        
        if select_mode == "file":
            file = h5_handling.simple_load(self._path_file)
            ## Check that the file has the correct format
            assert "mask_images" in file, "FR ERROR: 'mask_images' not found in file."
            self.mask_images = file["mask_images"]
            ## Check that the mask images have the correct format
            assert isinstance(self.mask_images, (dict,)), "FR ERROR: 'mask_images' must be a dict containing boolean numpy arrays representing the mask images."
            assert all([isinstance(mask, np.ndarray) for mask in self.mask_images.values()]), "FR ERROR: 'mask_images' from file is expected to be a 3D or list of 2D boolean np.ndarray."
            assert all([mask.shape == self.mask_images[list(self.mask_images.keys())[0]].shape for mask in self.mask_images.values()]), "FR ERROR: 'mask_images' must all have the same shape."
            assert all([mask.dtype == bool for mask in self.mask_images.values()]), "FR ERROR: 'mask_images' must be boolean."
            self.mask_images = {k: np.array(v, dtype=np.bool_) for k, v in self.mask_images.items()}  ## Ensure that the masks are boolean np arrays
            ## Check that roi_points has the correct format
            assert "roi_points" in file, "FR ERROR: 'roi_points' not found in file."
            self.roi_points = file["roi_points"]
            self.roi_points = {k: np.array(v, dtype=np.float32) for k, v in self.roi_points.items()}  ## Ensure that the roi_points are float np arrays
            ## Check that exampleImage has the correct format
            assert "exampleImage" in file, "FR ERROR: 'exampleImage' not found in file."
            self.exampleImage = file["exampleImage"]
            ## Ensure that the exampleImage is a float np array
            self.exampleImage = np.array(self.exampleImage)
            if self.exampleImage.dtype == np.uint8:
                self.exampleImage = self.exampleImage.astype(np.float32) / 255.0
            self.img_hw = self.exampleImage.shape
            self.set_point_positions(point_positions=file['point_positions']) if file['point_positions'] is not None else None
            
        elif select_mode == "custom":
            print(f"FR: Initializing ROIs from points...") if self._verbose > 1 else None
            self.mask_images = _Select_ROI._compute_mask_frames(
                selected_points=self.roi_points,
                exampleImage=self.exampleImage,
            )
            self.set_point_positions(point_positions) if point_positions is not None else None
        
        self._fill_config_runInfo_runData(
            path_file=self._path_file,
            coords_rois=self.roi_points,
            point_positions=self.point_positions,
            mask_images=self.mask_images,
        )

    def _fill_config_runInfo_runData(
        self,
        path_file=None,
        coords_rois=None,
        point_positions=None,
        mask_images=None,
    ):
        ## For FR_Module compatibility
        self.config = {
            "select_mode": self._select_mode,
            "exampleImage": (self.exampleImage is not None),
            "path_file": path_file,
            "roi_points": (coords_rois is not None),
            "point_positions": (point_positions is not None),
            "mask_images": (mask_images is not None),
            "verbose": self._verbose,
        }
        self.run_info = {
            "img_hw": self.img_hw,
        }
        self.run_data = {
            "mask_images": self.mask_images,
            "roi_points": self.roi_points,
            "point_positions": self.point_positions,
            "exampleImage": self.exampleImage,
        }

    def make_points(self, rois, point_spacing=10):
        """
        Generates a regular grid of tracking points inside the intersection of
        the supplied ROI masks and stores them on ``self.point_positions``.

        Args:
            rois (Union[List[np.ndarray], np.ndarray]):
                Either a list of 2D boolean masks, a single 2D boolean mask,
                or a 3D boolean array stacked along axis 0. All masks must
                share the same shape.
            point_spacing (int):
                Spacing between adjacent grid points, in pixels. (Default is
                ``10``)
        """
        ## Assertions
        ## rois should either be a list of 2D arrays or 3D array or a single 2D array
        assert isinstance(rois, (list, np.ndarray)), "FR ERROR: 'rois' must be a list of 2D arrays or a 3D array or a single 2D array."
        if isinstance(rois, list):
            assert all([isinstance(roi, np.ndarray) for roi in rois]), "FR ERROR: 'rois' must be a list of 2D arrays or a 3D array or a single 2D array."
            assert all([roi.shape == rois[0].shape for roi in rois]), "FR ERROR: shapes of all 'rois' must be the same."
            assert all([roi.dtype == bool for roi in rois]), "FR ERROR: 'rois' must be boolean."
        elif isinstance(rois, np.ndarray):
            if rois.ndim == 2:
                assert rois.dtype == bool, "FR ERROR: 'rois' must be boolean."
                rois = [rois]
            elif rois.ndim == 3:
                assert all([roi.dtype == bool for roi in rois]), "FR ERROR: 'rois' must be boolean."
                rois = [roi for roi in rois]

        ## Make points within rois_points with spacing of point_spacing
        ##  First make a single ROI boolean image, then make points
        print("FR: Making points to track") if self._verbose > 1 else None
        rois_all = np.stack(rois, axis=0).all(axis=0)
        self.point_positions = self._helper_make_points(rois_all, point_spacing)
        self.num_points = self.point_positions.shape[0]
        print(f"FR: {self.point_positions.shape[0]} points total") if self._verbose > 1 else None

        self.config.update({
            "point_spacing": point_spacing,
        })
        self.run_data.update({
            "point_positions": self.point_positions,
            "num_points": self.num_points,
        })

    def _helper_make_points(self, roi, point_spacing):
        """
        Builds an evenly spaced grid of points lying inside a single boolean
        ROI mask.

        Args:
            roi (np.ndarray):
                2D boolean array where ``True`` marks pixels inside the ROI.
                shape: *(H, W)*, dtype: *bool*.
            point_spacing (int):
                Spacing between adjacent grid points, in pixels. Rounded to
                the nearest integer if a non-integer value is supplied.

        Returns:
            (np.ndarray):
                points (np.ndarray):
                    Point coordinates as ``(x, y)`` pairs. shape:
                    *(n_points, 2)*, dtype: *float32*.
        """
        ## Assert that roi is a 2D array of booleans
        assert isinstance(roi, np.ndarray), "FR ERROR: roi must be a numpy array"
        assert roi.ndim == 2, "FR ERROR: roi must be a 2D array"
        assert roi.dtype == bool, "FR ERROR: roi must be a 2D array of booleans"
        ## Warn if point_spacing is not an integer. It will be rounded.
        if not isinstance(point_spacing, int):
            print("FR WARNING: point_spacing must be an integer. It will be rounded.")
            point_spacing = int(round(point_spacing))

        ## make point cloud
        y, x = np.where(roi)
        y_min, y_max = y.min(), y.max()
        x_min, x_max = x.min(), x.max()
        y_points = np.arange(y_min, y_max, point_spacing)
        x_points = np.arange(x_min, x_max, point_spacing)
        y_points, x_points = np.meshgrid(y_points, x_points)
        y_points = y_points.flatten()
        x_points = x_points.flatten()
        ## remove points outside of roi
        points = np.stack([y_points, x_points], axis=1)
        points = points[roi[points[:, 0], points[:, 1]]].astype(np.float32)
        ## flip to (x,y)
        points = np.fliplr(points)

        return points
    
    def set_point_positions(self, point_positions):
        """
        Manually overrides ``self.point_positions`` with an explicit array.

        Args:
            point_positions (np.ndarray):
                Tracking point coordinates as ``(x, y)`` pairs. shape:
                *(n_points, 2)*.
        """
        assert isinstance(point_positions, np.ndarray), "FR ERROR: 'point_positions' must be a numpy array."
        assert point_positions.ndim == 2, "FR ERROR: 'point_positions' must be a 2D array."
        assert point_positions.shape[1] == 2, "FR ERROR: 'point_positions' must have shape (n_points, 2)."
        
        self.point_positions = point_positions
        self.num_points = self.point_positions.shape[0]

    def __repr__(self):
        return f"ROIs object. Select mode: '{self._select_mode}'. Number of ROIs: {len(self.mask_images)}."

    ## Define methods for loading and handling videos
    def __getitem__(self, index): 
        if isinstance(index, int):
            index = list(self.mask_images.keys())[index]
        return self.mask_images[index]
    def __len__(self): return len(self.maks_images)
    def __iter__(self): return iter(self.mask_images)
    def __next__(self): return next(self.mask_images)

    def plot_rois(self, image=None, **kwargs_imshow):
        """
        Plots ROI polygon outlines (and tracking points if available) on top
        of an image.

        Args:
            image (np.ndarray):
                Background image to draw the ROIs on. If ``None``, falls back
                to ``self.exampleImage``; if that is also missing, a blank
                image is used. (Default is ``None``)
            **kwargs_imshow:
                Additional keyword arguments forwarded to
                ``matplotlib.pyplot.imshow``.

        Returns:
            (tuple): tuple containing:
                fig (matplotlib.figure.Figure):
                    The Matplotlib figure containing the plot.
                ax (matplotlib.axes.Axes):
                    The Matplotlib axes containing the plot.
        """
        import matplotlib.pyplot as plt
        if image is None:
            if hasattr(self, "exampleImage"):
                image = self.exampleImage
            else:
                print("FR WARNING: self.exampleImage does not exist. Plotting a blank image.")
                image = np.zeros((self.img_hw[0], self.img_hw[1]), dtype=np.uint8)


        ## set backend to non-interactive
        fig, ax = plt.subplots(1, 1)
        ax.imshow(image, **kwargs_imshow)
        ## Make mask polygons
        for ii, mask in enumerate(self.mask_images.values()):
            ax.contour(mask, colors=[plt.cm.tab20(ii)], linewidths=2, alpha=0.5)
        ## Show points on the image
        if self.point_positions is not None:
            ax.scatter(
                self.point_positions[:, 0], 
                self.point_positions[:, 1], 
                s=2, 
                color="red",
                alpha=0.5,
            )
        ## show figure
        plt.show()
        return fig, ax
    
    def fliplr(self):
        """
        Flips the example image, masks, ROI polygon points, and tracking
        points horizontally **in place**.
        """
        if hasattr(self, 'exampleImage'):
            if self.exampleImage is not None:
                self.exampleImage = np.fliplr(self.exampleImage)

        if hasattr(self, 'mask_images'):
            if self.mask_images is not None:
                self.mask_images = {k: np.fliplr(m) for k, m in self.mask_images.items()}
                
        if hasattr(self, 'roi_points'):
            if self.roi_points is not None:
                for k, p in self.roi_points.items():
                    self.roi_points[k][:, 0] = self.img_hw[1] - p[:, 0]
                    
        if hasattr(self, 'point_positions'):
            if self.point_positions is not None:
                self.point_positions[:, 0] = self.img_hw[1] - self.point_positions[:, 0]

        self._fill_config_runInfo_runData(
            path_file=self._path_file,
            coords_rois=self.roi_points,
            point_positions=self.point_positions,
            mask_images=self.mask_images,
        )

class _Select_ROI:
    """
    Interactive polygon ROI selector built on Plotly ``FigureWidget`` and
    ipywidgets. Replaces the original matplotlib/nbagg GUI and works in
    JupyterLab 3+, Notebook 7+, VSCode Jupyter, and any frontend that supports
    ``FigureWidget`` (anywidget backend, requires ``plotly >= 6.0``).

    Draw closed polygons on the displayed image via the ``drawclosedpath``
    modebar tool, then click "Confirm ROIs" to freeze the selection and
    populate the output attributes. Colab note: ``plotly >= 6.0`` is broken
    on Colab (anywidget JS issue #5027); workaround is ``plotly==5.24.1`` plus
    ``from google.colab import output; output.enable_custom_widget_manager()``.

    Args:
        image (np.ndarray):
            Image to display in the selector. shape: *(H, W)* or *(H, W, 3)*.
        n_rois (int):
            Hint for the title text indicating how many polygons to draw.
            (Default is ``1``)
        height (Optional[int]):
            Optional figure height in pixels. (Default is ``None``)
        width (Optional[int]):
            Optional figure width in pixels. (Default is ``None``)
        line_color (str):
            Stroke color used while drawing new polygons. (Default is
            ``'red'``)

    Attributes:
        selected_points (dict):
            Populated after "Confirm ROIs" is clicked. Maps ``"ROI_0"``,
            ``"ROI_1"``, ... to ``np.ndarray`` of shape *(N, 2)*, dtype
            *float64*, columns ``(x, y)`` in image-pixel coords (top-left
            origin; ``x`` is column, ``y`` is row).
        mask_frames (dict):
            Populated after "Confirm ROIs" is clicked. Maps ``"mask_0"``,
            ``"mask_1"``, ... to a boolean ``np.ndarray`` of shape *(H, W)*.
            ``True`` inside the polygon.
        _completed_status (bool):
            ``True`` after "Confirm ROIs" has been clicked successfully.
    """

    _ROI_COLORS: list = [
        "red", "blue", "lime", "orange", "purple",
        "cyan", "magenta", "yellow",
    ]

    def __init__(
        self,
        image: np.ndarray,
        n_rois: int = 1,
        height=None,
        width=None,
        line_color: str = "red",
    ) -> None:
        """Builds the FigureWidget, wires up button callbacks, and renders the GUI."""
        import re as _re
        import plotly.express as px
        import plotly.graph_objects as go
        import ipywidgets as widgets
        from IPython.display import display
        import skimage.draw

        self._re = _re
        self._skimage_draw = skimage.draw

        assert isinstance(image, np.ndarray), (
            "FR ERROR: '_Select_ROI' 'image' must be a numpy ndarray, "
            f"got {type(image)}."
        )
        assert image.ndim in (2, 3), (
            "FR ERROR: '_Select_ROI' 'image' must be 2D (H,W) or 3D (H,W,3), "
            f"got shape {image.shape}."
        )

        self._image = image

        ## Public attributes — initialised as empty dicts so that ROIs.__init__
        ## can grab references before the user clicks Confirm.  _on_confirm
        ## mutates them in place so the reference-sharers stay in sync.
        self.selected_points: dict = {}
        self.mask_frames: dict = {}
        self._completed_status: bool = False

        ## Build the base figure using px.imshow.
        ## binary_string=True PNG-encodes the array for fast rendering and
        ## sets up a top-left origin (y=0 at top), matching NumPy indexing.
        if image.ndim == 2:
            fig_base = px.imshow(
                image,
                color_continuous_scale="gray",
                binary_string=True,
            )
        else:
            fig_base = px.imshow(image, binary_string=True)

        layout_kwargs = dict(
            dragmode="drawclosedpath",
            newshape=dict(
                line_color=line_color,
                fillcolor="rgba(255,0,0,0.2)",
                ## opacity > 0.5 required to click inside shape to select it
                opacity=0.6,
            ),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
            margin=dict(l=0, r=0, t=30, b=0),
            title_text=f"Draw {n_rois} ROI polygon(s) → click Confirm ROIs",
            coloraxis_showscale=False,
        )
        if height is not None:
            layout_kwargs["height"] = height
        if width is not None:
            layout_kwargs["width"] = width

        fig_base.update_layout(**layout_kwargs)

        ## FigureWidget (anywidget-backed in plotly >= 6) enables Python-side
        ## shape retrieval. Must display via IPython.display, NOT .show().
        self._widget = go.FigureWidget(fig_base)

        self._btn_confirm = widgets.Button(
            description="Confirm ROIs",
            button_style="success",
            tooltip="Freeze current polygon selections and compute masks",
        )
        self._btn_clear = widgets.Button(
            description="Clear",
            button_style="warning",
            tooltip="Remove all drawn shapes from the figure",
        )
        self._label = widgets.Label("Draw polygons using the toolbar, then click Confirm ROIs.")

        self._btn_confirm.on_click(self._on_confirm)
        self._btn_clear.on_click(self._on_clear)

        self._label.layout = widgets.Layout(width='100%')
        display(
            widgets.VBox([
                self._widget,
                widgets.HBox([self._btn_confirm, self._btn_clear]),
                self._label,
            ])
        )

    def _on_confirm(self, _button_event) -> None:
        """
        Reads shapes from the FigureWidget, parses each SVG path into polygon
        vertices, computes the corresponding boolean masks, and updates
        ``self.selected_points`` and ``self.mask_frames`` in place.

        Args:
            _button_event (object):
                Unused ipywidgets button event passed by the click callback.
        """
        ## Use _props workaround: widget.layout["shapes"] returns empty tuple
        ## due to plotly >= 6.x bug (plotly/plotly.py #5309, open Aug 2025).
        shapes_raw = self._widget.layout._props.get("shapes", [])

        path_shapes = [s for s in shapes_raw if s.get("type") == "path"]

        if not path_shapes:
            self._label.value = "No polygon shapes found. Draw at least one polygon."
            return

        selected_points: dict = {}
        mask_frames: dict = {}

        for idx, shape in enumerate(path_shapes):
            path_str = shape["path"]
            verts = self._parse_svg_path(path_str)
            mask = self._compute_mask(
                vertices_xy=verts,
                image_shape=self._image.shape,
            )
            selected_points[f"ROI_{idx}"] = verts
            mask_frames[f"mask_{idx}"] = mask

        ## Mutate in place so that reference-sharers (ROIs.roi_points,
        ## ROIs.mask_images) stay in sync after the user clicks Confirm.
        self.selected_points.clear()
        self.selected_points.update(selected_points)
        self.mask_frames.clear()
        self.mask_frames.update(mask_frames)
        self._completed_status = True

        n = len(selected_points)
        vertex_counts = [v.shape[0] for v in selected_points.values()]
        self._label.value = (
            f"Captured {n} ROI(s) with {vertex_counts} vertices. "
            "Access via .selected_points and .mask_frames."
        )
        print(
            f"Select_ROI_Plotly: Captured {n} ROI(s) with "
            f"{vertex_counts} vertices."
        )

    def _on_clear(self, _button_event) -> None:
        """
        Removes all drawn shapes from the figure and resets the internal
        ``_props['shapes']`` cache so the next confirm sees an empty list.

        Args:
            _button_event (object):
                Unused ipywidgets button event passed by the click callback.
        """
        ## Bug workaround (plotly #5309): writing widget.layout.shapes = []
        ## inside batch_update() doesn't push the clear to the JS frontend.
        ## Use plotly_relayout to send the update directly; also reset _props
        ## so that the next Confirm read sees an empty list.
        self._widget.plotly_relayout({'shapes': []})
        self._widget.layout._props['shapes'] = []
        self._label.value = "Shapes cleared. Draw new polygons."

    @staticmethod
    def _parse_svg_path(path_str: str) -> np.ndarray:
        """
        Parses a Plotly ``drawclosedpath`` SVG path string into polygon
        vertices.

        Handles the format Plotly actually emits (``Mx,yLx,yLx,yZ`` with
        absolute ``M`` and ``L`` commands, comma-separated ``x,y`` pairs, and
        a trailing ``Z`` close), and also accepts space-separated variants.

        Args:
            path_str (str):
                SVG path string from
                ``widget.layout._props['shapes'][i]['path']``.

        Returns:
            (np.ndarray):
                vertices (np.ndarray):
                    Polygon vertices as ``(x, y)`` pairs. shape: *(N, 2)*,
                    dtype: *float64*.

        Raises:
            ValueError:
                If ``path_str`` is empty or yields no vertices.
        """
        import re
        if not path_str or not path_str.strip():
            raise ValueError("Empty SVG path string.")

        tokens = re.split(r"([MLZmlz])", path_str.strip())
        tokens = [t.strip() for t in tokens if t.strip()]

        coords: list = []
        current_pos = np.array([0.0, 0.0])

        i = 0
        while i < len(tokens):
            cmd = tokens[i]
            i += 1

            if cmd in ("M", "L"):
                if i < len(tokens) and tokens[i] not in "MLZmlz":
                    raw = tokens[i].replace(",", " ").split()
                    i += 1
                    for j in range(0, len(raw) - 1, 2):
                        x, y = float(raw[j]), float(raw[j + 1])
                        current_pos = np.array([x, y])
                        coords.append(current_pos.copy())

            elif cmd in ("m", "l"):
                if i < len(tokens) and tokens[i] not in "MLZmlz":
                    raw = tokens[i].replace(",", " ").split()
                    i += 1
                    for j in range(0, len(raw) - 1, 2):
                        dx, dy = float(raw[j]), float(raw[j + 1])
                        current_pos = current_pos + np.array([dx, dy])
                        coords.append(current_pos.copy())

            elif cmd in ("Z", "z"):
                pass

        if not coords:
            raise ValueError(f"No vertices parsed from path: {path_str!r}")

        return np.array(coords, dtype=np.float64)

    @staticmethod
    def _compute_mask_frames(
        selected_points: dict,
        exampleImage: np.ndarray,
        verbose: bool = False,
    ) -> dict:
        """
        Batch-computes boolean masks from a dictionary of polygon vertex
        arrays.

        Args:
            selected_points (dict):
                Maps ``"ROI_0"``, ... to an ``np.ndarray`` of shape *(N, 2)*
                with columns ``(x, y)``.
            exampleImage (np.ndarray):
                Image whose ``(H, W)`` determines the mask dimensions.
            verbose (bool):
                If ``True``, print a confirmation line when done. (Default is
                ``False``)

        Returns:
            (dict):
                mask_frames (dict):
                    Maps ``"mask_0"``, ... to a boolean ``np.ndarray`` of
                    shape *(H, W)*.
        """
        import skimage.draw
        mask_frames: dict = {}
        for ii, pts in enumerate(selected_points.values()):
            pts = np.array(pts)
            mask_frame = np.zeros((exampleImage.shape[0], exampleImage.shape[1]))
            rr, cc = skimage.draw.polygon(pts[:, 1], pts[:, 0])
            mask_frame[rr, cc] = 1
            mask_frame = mask_frame.astype(np.bool_)
            mask_frames[f"mask_{ii}"] = mask_frame
        if verbose:
            print("mask_frames computed")
        return mask_frames

    @staticmethod
    def _compute_mask(
        vertices_xy: np.ndarray,
        image_shape: tuple,
    ) -> np.ndarray:
        """
        Converts a single set of polygon vertices to a boolean image mask.

        Args:
            vertices_xy (np.ndarray):
                Polygon vertices as ``(x, y)`` pairs in pixel coordinates.
                shape: *(N, 2)*.
            image_shape (tuple):
                Either ``(height, width)`` or ``(height, width, channels)``.

        Returns:
            (np.ndarray):
                mask (np.ndarray):
                    Boolean polygon mask. shape: *(height, width)*, dtype:
                    *bool*.
        """
        import skimage.draw
        h, w = image_shape[:2]
        rows = vertices_xy[:, 1]
        cols = vertices_xy[:, 0]
        rr, cc = skimage.draw.polygon(rows, cols, shape=(h, w))
        mask = np.zeros((h, w), dtype=np.bool_)
        mask[rr, cc] = True
        return mask
        
        

##########################################################################################################################################
######################################################### MULTISESSION ALIGNMENT #########################################################
##########################################################################################################################################


class ROI_Alinger:
    """
    Registers a template image to a set of new images using OpenCV optical
    flow, then warps the template's ROI polygons and tracking points onto
    each new image. RH 2022

    Args:
        method (str):
            Optical-flow method to use for non-rigid registration. One of \n
            * ``'calcOpticalFlowFarneback'``
            * ``'createOptFlow_DeepFlow'`` \n
            (Default is ``'createOptFlow_DeepFlow'``)
        kwargs_method (dict):
            Keyword arguments forwarded to the chosen optical-flow method.
            If ``None``, hard-coded defaults are used. (Default is ``None``)
        verbose (int):
            Verbosity level. One of \n
            * ``0``: No updates.
            * ``1``: Warnings only.
            * ``2``: All updates. \n
            (Default is ``1``)
    """
    def __init__(
        self,
        method='createOptFlow_DeepFlow',
        kwargs_method=None,
        verbose=1,
    ):
        """Stores the chosen optical-flow method and its keyword arguments."""
        self._verbose = verbose
        self._method = method
        self._kwargs_method = kwargs_method

    def align_and_make_ROIs(
        self,
        ROIs_object_template,
        images_new,
        image_template=None,
        template_method='image',
        shifts=None,
        normalize=True,
    ):
        """
        Performs non-rigid registration of a template image onto each new
        image and warps the template's tracking points, ROI polygons, masks,
        and the new images themselves into the template's frame. RH 2022

        Results are stored on the instance as ``self.flows``,
        ``self.pointPositions_new``, ``self.roiPoints_new``,
        ``self.maskImages_new``, ``self.ROIs_objects_new``, and
        ``self.images_warped``.

        Args:
            ROIs_object_template (ROIs):
                A single ``ROIs`` object built from the template image. Its
                ROIs and tracking points are warped onto each new image.
            images_new (List[np.ndarray]):
                Images to align the template to. Each image must have shape
                *(H, W, n_channels)* and dtype *uint8*.
            image_template (np.ndarray):
                Template image to warp onto the new images. shape:
                *(H, W, n_channels)*, dtype: *uint8*. If ``None``,
                ``ROIs_object_template.exampleImage`` is used. (Default is
                ``None``)
            template_method (str):
                Strategy for choosing the template per registration. One of \n
                * ``'image'``: ``image_template`` is treated as a single
                  image.
                * ``'sequential'``: ``image_template`` is treated as the
                  integer index of the image to use as the zero-offset
                  reference. \n
                (Default is ``'image'``)
            shifts (np.ndarray):
                Per-image ``(dx, dy)`` shifts to add to each computed flow
                field, e.g. from a phase-correlation pre-registration step.
                If ``None``, zero shifts are applied. (Default is ``None``)
            normalize (bool):
                If ``True``, normalize images to ``[0, 255]`` (using each
                image's own min and max) before registration. (Default is
                ``True``)
        """
        ### Assert images_new is a list of 2D or 3D numpy.ndarray
        if isinstance(images_new, list):
            assert all([isinstance(ii, np.ndarray) for ii in images_new]), f'Images_new must be a list of numpy.ndarray. Found {[type(im) for im in images_new]}'
            assert all([len(ii.shape) == 3 for ii in images_new]), 'Images_new must be a list of 3D numpy.ndarray'
        if isinstance(images_new, np.ndarray):
            assert len(images_new.shape) == 4, 'images_new must be a list of 3D numpy.ndarray'
            images_new = [images_new[ii] for ii in range(images_new.shape[0])]
        ### Assert template_method is a string in ['image', 'sequential']
        assert isinstance(template_method, str), 'template_method must be a string'
        assert template_method in ['image', 'sequential'], 'template_method must be a string in ["image", "sequential"]'
        
        self._image_template = image_template if image_template is not None else ROIs_object_template.exampleImage
        self._images_new = images_new
        self._template_method = template_method
        self._normalize = normalize

        self._shifts = [(0,0)] * len(self._images_new) if shifts is None else shifts

        self._pointPositions_template = ROIs_object_template.point_positions ## List of point positions for each ROI
        self._roiPoints_template = ROIs_object_template.roi_points  ## List of points describing the outline of each ROI

        ## Make grid of indices for image remapping
        self._dims = self._image_template.shape
        self._x_arange, self._y_arange = np.arange(0., self._dims[1]).astype(np.float32), np.arange(0., self._dims[0]).astype(np.float32)
        self._x_grid, self._y_grid = np.meshgrid(self._x_arange, self._y_arange)

        ## Register images
        print('Registering images...')
        self.flows = [self._register_image(
            image_moving=self._image_template,
            image_template=im_new,
            shifts=shift,
            normalize=self._normalize,
        ) for im_new, shift in tqdm(zip(self._images_new, self._shifts), total=len(self._images_new))]

        ## Warp point_positions
        print('Warping point positions...')
        self.pointPositions_new = [self._warp_points(
            points=self._pointPositions_template,
            flow=flow,
        ) for flow in tqdm(self.flows)]

        ## Warp ROI outlines
        print('Warping ROI outlines...')
        self.roiPoints_new = [{
            key: self._warp_points(
                points=points,
                flow=flow,
            ) for key,points in self._roiPoints_template.items()} for flow in tqdm(self.flows)]

        ## Make mask images
        print('Making mask images...')
        self.maskImages_new = [_Select_ROI._compute_mask_frames(
            selected_points=points,
            exampleImage=self._image_template,
        ) for points in tqdm(self.roiPoints_new)]

        ## Make ROIs object
        print('Making ROIs objects...')
        self.ROIs_objects_new = [ROIs(
            select_mode='custom',
            exampleImage=self._image_template,
            points=points,
        ) for points in tqdm(self.roiPoints_new)]

        ## Warp images
        print('Warping images...')
        self.images_warped = [self._warp_image(
            image=img,
            flow=flow,
        ) for flow,img in tqdm(zip(self.flows, self._images_new), total=len(self._images_new))]

    def _register_image(
        self,
        image_moving,
        image_template,
        shifts=None,
        normalize=True,
    ):
        """
        Computes the dense optical-flow field that registers ``image_moving``
        onto ``image_template`` using the configured OpenCV method. RH 2022

        Args:
            image_moving (np.ndarray):
                Image to align onto ``image_template``. shape: *(H, W)* or
                *(H, W, n_channels)*, dtype: *uint8*.
            image_template (np.ndarray):
                Reference image to align onto. shape: *(H, W)* or
                *(H, W, n_channels)*, dtype: *uint8*.
            shifts (Union[np.ndarray, tuple]):
                Optional ``(dx, dy)`` shift (or array of shifts) added to the
                returned flow field. (Default is ``None``)
            normalize (bool):
                If ``True``, normalize both inputs to ``[0, 255]`` before
                registration. (Default is ``True``)

        Returns:
            (np.ndarray):
                flow (np.ndarray):
                    Dense optical-flow field mapping ``image_moving`` to
                    ``image_template``. shape: *(H, W, 2)*, last dim is
                    ``(dx, dy)``.
        """

        # Check inputs
        ### Assert image_template is a 3D numpy.ndarray
        assert isinstance(image_template, np.ndarray), 'image_template must be a numpy.ndarray'
        assert image_template.ndim in [2,3], 'image_template must be a 2D or 3D numpy.ndarray'
        if image_template.ndim == 3:
            image_template = image_template.mean(axis=2)
        ### Assert image_moving is a 2D or 3D numpy.ndarray
        assert isinstance(image_moving, np.ndarray), 'image_moving must be a numpy.ndarray'
        assert image_moving.ndim in [2, 3], 'image_moving must be a 2D or 3D numpy.ndarray'
        if image_moving.ndim == 3:
            image_moving = image_moving.mean(axis=2)
        ### Assert shifts is a numpy.ndarray or None or tuple of length 2
        assert isinstance(shifts, (np.ndarray, type(None), tuple)), 'shifts must be a numpy.ndarray or None or tuple of length 2'
        if isinstance(shifts, np.ndarray):
            assert len(shifts.shape) == 2, 'shifts must be a 2D numpy.ndarray'
            assert shifts.shape[1] == 2, 'shifts must be of shape (n_images, 2)'
        if isinstance(shifts, tuple):
            assert len(shifts) == 2, 'shifts must be a tuple of length 2'
            shifts = np.array(shifts)

        # Normalize images
        if normalize:
            image_moving = ((image_moving - image_moving.min()) / (image_moving.max() - image_moving.min()) * 255).astype(np.uint8)
            image_template = ((image_template - image_template.min()) / (image_template.max() - image_template.min()) * 255).astype(np.uint8)
        
        if self._method == 'calcOpticalFlowFarneback':
            if self._kwargs_method is None:
                self._kwargs_method = {
                    'pyr_scale': 0.3, 
                    'levels': 3,
                    'winsize': 128, 
                    'iterations': 7,
                    'poly_n': 7, 
                    'poly_sigma': 1.5,
                    'flags': cv2.OPTFLOW_FARNEBACK_GAUSSIAN
                }
            flow = cv2.calcOpticalFlowFarneback(
                prev=image_moving,
                next=image_template, 
                flow=None, 
                **self._kwargs_method,
            )
    
        elif self._method == 'createOptFlow_DeepFlow':
            flow = cv2.optflow.createOptFlow_DeepFlow().calc(
                image_moving,
                image_template,
                None
            )

        ### Apply shifts
        if shifts is not None:
            flow += shifts
            
        return flow 


    def _warp_points(
        self,
        points,
        flow,
    ):
        """
        Warps a set of ``(x, y)`` points using a dense flow field. RH 2022

        Args:
            points (np.ndarray):
                Points to warp as ``(x, y)`` pairs. shape: *(n_points, 2)*,
                dtype: *float*.
            flow (np.ndarray):
                Dense flow field. shape: *(H, W, 2)*, dtype: *float*. Last
                dim is ``(dx, dy)``.

        Returns:
            (np.ndarray):
                points_remap (np.ndarray):
                    Warped points clipped to the image bounds. shape:
                    *(n_points, 2)*.
        """
        from functools import partial
        ### Assert points is a 2D numpy.ndarray of shape (n_points, 2) and that all points are within the image and that points are float
        assert isinstance(points, np.ndarray), 'points must be a numpy.ndarray'
        assert len(points.shape) == 2, 'points must be a 2D numpy.ndarray'
        assert points.shape[1] == 2, 'points must be of shape (n_points, 2)'
        assert (points[:,0] >= 0).all(), 'points must be within the image'
        assert (points[:,0] < flow.shape[1]).all(), 'points must be within the image'
        assert (points[:,1] >= 0).all(), 'points must be within the image'
        assert (points[:,1] < flow.shape[0]).all(), 'points must be within the image'
        assert np.issubdtype(points.dtype, np.floating), 'points must be a float subtype'

        x_remap = (self._x_grid + flow[:, :, 0]).astype(np.float32)
        y_remap = (self._y_grid + flow[:, :, 1]).astype(np.float32)

        ## Use a RectBivariateSpline to remap points
        splineGrid = partial(
            scipy.interpolate.RectBivariateSpline,
            x=self._y_arange,
            y=self._x_arange,
            kx=1,
            ky=1,
            s=0
        )
        splineGrid_x, splineGrid_y = splineGrid(z=x_remap), splineGrid(z=y_remap)
        points_remap_x = splineGrid_x.ev(
            points[:, 1],
            points[:, 0],
        )
        points_remap_y = splineGrid_y.ev(
            points[:, 1],
            points[:, 0],
        )

        points_remap = np.array([points_remap_x, points_remap_y]).T

        ## Clip points to image size
        points_remap[:, 0] = np.clip(points_remap[:, 0], 0, flow.shape[1] - 1)
        points_remap[:, 1] = np.clip(points_remap[:, 1], 0, flow.shape[0] - 1)

        return points_remap


    def _warp_image(
        self,
        image,
        flow,
    ):
        """
        Warps an image using a dense flow field via ``cv2.remap``. RH 2022

        Args:
            image (np.ndarray):
                Image to warp. shape: *(H, W)* or *(H, W, 3)*, dtype:
                *float*. 3-channel inputs are averaged to grayscale before
                remapping.
            flow (np.ndarray):
                Dense flow field. shape: *(H, W, 2)*, dtype: *float*. Last
                dim is ``(dx, dy)``.

        Returns:
            (np.ndarray):
                image_remap (np.ndarray):
                    Warped image. shape: *(H, W)*, dtype: *float32*.
        """
        def safe_remap(image, x_remap, y_remap):
            image_remap = cv2.remap(
                image.astype(np.float32),
                x_remap,
                y_remap, 
                cv2.INTER_LINEAR
            )
            if image_remap.sum() == 0:
                image_remap = image
            return image_remap

        if image.ndim == 3:
            image = image.mean(axis=2)
        assert image.ndim == 2, 'image must be 2D'
        assert image.shape[0] == flow.shape[0], 'image and flow must have the same height'
        assert image.shape[1] == flow.shape[1], 'image and flow must have the same width'
        assert np.issubdtype(image.dtype, np.floating), 'image must be a float subtype'

        x_remap = (self._x_grid + flow[:, :, 0]).astype(np.float32)
        y_remap = (self._y_grid + flow[:, :, 1]).astype(np.float32)

        image_remap = safe_remap(
            image=image,
            x_remap=x_remap,
            y_remap=y_remap
        )

        return image_remap



from typing import List, Optional, Tuple, Union

class Image_Aligner(FR_Module):
    """
    A class for registering points to a template image. Currently relies on
    available OpenCV methods for rigid and registration.
    RH 2023

    Args:
        verbose (bool):
            Whether to print progress updates. (Default is ``True``)
    """
    def __init__(
        self,
        verbose=True,
    ):
        """Initializes empty registration state (warp matrices and remap indices)."""
        self._verbose = verbose

        self.remappingIdx_geo = None
        self.warp_matrices = None

        self.remappingIdx_nonrigid = None

        self._HW = None

    @classmethod
    def augment_images(
        cls,
        ims: List[np.ndarray],
        use_CLAHE: bool = True,
        CLAHE_grid_size: int = 1,
        CLAHE_clipLimit: int = 1,
        CLAHE_normalize: bool = True,
    ) -> None:
        """
        Augments the FOV images by mixing the FOV with the ROI images and
        optionally applying CLAHE.
        RH 2023

        Args:
            ims (List[np.ndarray]): 
                A list of FOV images.
            use_CLAHE (bool):
                Whether to apply CLAHE to the images. (Default is ``True``)
            CLAHE_grid_size (int):
                The grid size for CLAHE. See alignment.clahe for more details.
                (Default is *1*)
            CLAHE_clipLimit (int):
                The clip limit for CLAHE. See alignment.clahe for more details.
                (Default is *1*)
            CLAHE_normalize (bool):
                Whether to normalize the CLAHE output. See alignment.clahe for
                more details. (Default is ``True``)

        Returns:
            (List[np.ndarray]):
                FOV_images_augmented (List[np.ndarray]):
                    The augmented FOV images.
        """
        
        # h,w = ims[0].shape
        ims_aug = [clahe(im, grid_size=CLAHE_grid_size, clipLimit=CLAHE_clipLimit, normalize=CLAHE_normalize) for im in tqdm(ims)] if use_CLAHE else ims

        return ims_aug

    def fit_geometric(
        self,
        template: Union[int, np.ndarray],
        ims_moving: List[np.ndarray],
        template_method: str = 'sequential',
        mode_transform: str = 'affine',
        gaussFiltSize: int = 11,
        mask_borders: Tuple[int, int, int, int] = (0, 0, 0, 0),
        n_iter: int = 1000,
        termination_eps: float = 1e-9,
        auto_fix_gaussFilt_step: Union[bool, int] = 10,
    ) -> np.ndarray:
        """
        Performs geometric registration of ``ims_moving`` to a template, using 
        ``cv2.findTransformECC``. 
        RH 2023

        Args:
            template (Union[int, np.ndarray]): 
                Depends on the value of 'template_method'. 
                If 'template_method' == 'image', this should be a 2D np.ndarray image, an integer index 
                of the image to use as the template, or a float between 0 and 1 representing the fractional 
                index of the image to use as the template. 
                If 'template_method' == 'sequential', then template is the integer index or fractional index 
                of the image to use as the template.
            ims_moving (List[np.ndarray]): 
                List of images to be aligned.
            template_method (str): 
                Method to use for template selection. \n
                * 'image': use the image specified by 'template'. 
                * 'sequential': register each image to the previous or next image \n
                (Default is 'sequential')
            mode_transform (str): 
                Mode of geometric transformation. Can be 'translation', 'euclidean', 
                'affine', or 'homography'. See ``cv2.findTransformECC`` for more details. 
                (Default is 'affine')
            gaussFiltSize (int): 
                Size of the Gaussian filter. (Default is *11*)
            mask_borders (Tuple[int, int, int, int]): 
                Border mask for the image. Format is (top, bottom, left, right). 
                (Default is (0, 0, 0, 0))
            n_iter (int): 
                Number of iterations for ``cv2.findTransformECC``. (Default is *1000*)
            termination_eps (float): 
                Termination criteria for ``cv2.findTransformECC``. (Default is *1e-9*)
            auto_fix_gaussFilt_step (Union[bool, int]): 
                Automatically fixes convergence issues by increasing the gaussFiltSize. 
                If ``False``, no automatic fixing is performed. If ``True``, the 
                gaussFiltSize is increased by 2 until convergence. If int, the gaussFiltSize 
                is increased by this amount until convergence. (Default is *10*)

        Returns:
            (np.ndarray): 
                remapIdx_geo (np.ndarray): 
                    An array of shape *(N, H, W, 2)* representing the remap field for N images.
        """
        ## Imports
        super().__init__()
        
        # Check if ims_moving is a non-empty list
        assert len(ims_moving) > 0, "ims_moving must be a non-empty list of images."
        # Check if all images in ims_moving have the same shape
        shape = ims_moving[0].shape
        for im in ims_moving:
            assert im.shape == shape, "All images in ims_moving must have the same shape."
        # Check if template_method is valid
        valid_template_methods = {'sequential', 'image'}
        assert template_method in valid_template_methods, f"template_method must be one of {valid_template_methods}"
        # Check if mode_transform is valid
        valid_mode_transforms = {'translation', 'euclidean', 'affine', 'homography'}
        assert mode_transform in valid_mode_transforms, f"mode_transform must be one of {valid_mode_transforms}"
        # Check if gaussFiltSize is a number (float or int)
        assert isinstance(gaussFiltSize, (float, int)), "gaussFiltSize must be a number."
        # Convert gaussFiltSize to an odd integer
        gaussFiltSize = int(np.round(gaussFiltSize))

        H, W = ims_moving[0].shape[:2]
        self._HW = (H,W) if self._HW is None else self._HW

        ims_moving, template = self._fix_input_images(ims_moving=ims_moving, template=template, template_method=template_method)

        self.mask_geo = helpers.mask_image_border(
            im=np.ones((H, W), dtype=np.uint8),
            border_outer=mask_borders,
            mask_value=0,
        )

        print(f'Finding geometric registration warps with mode: {mode_transform}, template_method: {template_method}, mask_borders: {mask_borders is not None}') if self._verbose else None
        warp_matrices_raw = []
        for ii, im_moving in tqdm(enumerate(ims_moving), desc='Finding geometric registration warps', total=len(ims_moving), disable=not self._verbose):
            if template_method == 'sequential':
                ## warp images before template forward (t1->t2->t3->t4)
                if ii < template:
                    im_template = ims_moving[ii+1]
                ## warp template to itself
                elif ii == template:
                    im_template = ims_moving[ii]
                ## warp images after template backward (t4->t3->t2->t1)
                elif ii > template:
                    im_template = ims_moving[ii-1]
            elif template_method == 'image':
                im_template = template

            if im_template.ndim == 3:
                im_template = im_template.mean(2)
            if im_moving.ndim == 3:
                im_moving = im_moving.mean(2)
            
            def _safe_find_geometric_transformation(gaussFiltSize, attempt=0):
                if attempt >= 10:
                    raise Exception(f'Error finding geometric registration warp for image {ii}: Reached maximum number of attempts.')
                
                try:
                    warp_matrix = helpers.find_geometric_transformation(
                        im_template=im_template,
                        im_moving=im_moving,
                        warp_mode=mode_transform,
                        n_iter=n_iter,
                        mask=self.mask_geo,
                        termination_eps=termination_eps,
                        gaussFiltSize=gaussFiltSize,
                    )
                except Exception as e:
                    if auto_fix_gaussFilt_step:
                        print(f'Error finding geometric registration warp for image {ii}: {e}') if self._verbose else None
                        print(f'Increasing gaussFiltSize by {auto_fix_gaussFilt_step} to {gaussFiltSize + auto_fix_gaussFilt_step}') if self._verbose else None
                        return _safe_find_geometric_transformation(gaussFiltSize + auto_fix_gaussFilt_step, attempt=attempt+1)
                
                    print(f'Error finding geometric registration warp for image {ii}: {e}')
                    print(f'Defaulting to identity matrix warp.')
                    print(f'Consider doing one of the following:')
                    print(f'  - Make better images to input. You can add the spatialFootprints images to the FOV images to make them better.')
                    print(f'  - Increase the gaussFiltSize parameter. This will make the images blurrier, but may help with registration.')
                    print(f'  - Decrease the termination_eps parameter. This will make the registration less accurate, but may help with registration.')
                    print(f'  - Increase the mask_borders parameter. This will make the images smaller, but may help with registration.')
                    warp_matrix = np.eye(3)[:2,:]
                return warp_matrix
            
            warp_matrix = _safe_find_geometric_transformation(gaussFiltSize=gaussFiltSize)
            warp_matrices_raw.append(warp_matrix)

        
        # compose warp transforms
        print('Composing geometric warp matrices...') if self._verbose else None
        self.warp_matrices = []
        if template_method == 'sequential':
            ## compose warps before template forward (t1->t2->t3->t4)
            for ii in np.arange(0, template):
                warp_composed = self._compose_warps(
                    warp_0=warp_matrices_raw[ii], 
                    warps_to_add=warp_matrices_raw[ii+1:template+1],
                    warpMat_or_remapIdx='warpMat',
                )
                self.warp_matrices.append(warp_composed)
            ## compose template to itself
            self.warp_matrices.append(warp_matrices_raw[template])
            ## compose warps after template backward (t4->t3->t2->t1)
            for ii in np.arange(template+1, len(ims_moving)):
                warp_composed = self._compose_warps(
                    warp_0=warp_matrices_raw[ii], 
                    warps_to_add=warp_matrices_raw[template:ii][::-1],
                    warpMat_or_remapIdx='warpMat',
                )
                self.warp_matrices.append(warp_composed)
        ## no composition when template_method == 'image'
        elif template_method == 'image':
            self.warp_matrices = warp_matrices_raw

        self.warp_matrices = np.stack(self.warp_matrices, axis=0)

        # convert warp matrices to remap indices
        self.remappingIdx_geo = np.stack([helpers.warp_matrix_to_remappingIdx(warp_matrix=warp_matrix, x=W, y=H) for warp_matrix in self.warp_matrices], axis=0)

        return self.remappingIdx_geo   

    def fit_nonrigid(
        self,
        template: Union[int, np.ndarray],
        ims_moving: List[np.ndarray],
        remappingIdx_init: Optional[np.ndarray] = None,
        template_method: str = 'sequential',
        mode_transform: str = 'createOptFlow_DeepFlow',
        kwargs_mode_transform: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Perform geometric registration of ``ims_moving`` to a **template**.
        Currently relies on ``cv2.findTransformECC``.
        RH 2023

        Args:
            template (Union[int, np.ndarray]): 
                * If ``template_method`` == ``'image'``: Then **template** is
                  either an image or an integer index or a float fractional
                  index of the image to use as the **template**.
                * If ``template_method`` == ``'sequential'``: then **template**
                  is the integer index of the image to use as the **template**.
            ims_moving (List[np.ndarray]): 
                A list of images to be aligned.
            remappingIdx_init (Optional[np.ndarray]): 
                An array of shape *(N, H, W, 2)* representing any initial remap
                field to apply to the images in ``ims_moving``. The output of
                this method will be added/composed with ``remappingIdx_init``.
                (Default is ``None``)
            template_method (str): 
                The method to use for **template** selection. Either \n
                * ``'image'``: use the image specified by 'template'.
                * ``'sequential'``: register each image to the previous or next
                  image (will be next for images before the template and
                  previous for images after the template) \n
                (Default is 'sequential')
            mode_transform (str): 
                The type of transformation to use for registration. Either
                'createOptFlow_DeepFlow' or 'calcOpticalFlowFarneback'. (Default
                is 'createOptFlow_DeepFlow')
            kwargs_mode_transform (Optional[dict]): 
                Keyword arguments for the transform chosen. See cv2 docs for
                chosen transform. (Default is ``None``)

        Returns:
            (np.ndarray): 
                remapIdx_nonrigid (np.ndarray): 
                    An array of shape *(N, H, W, 2)* representing the remap
                    field for N images.
        """
        import cv2
        # Check if ims_moving is a non-empty list
        assert len(ims_moving) > 0, "ims_moving must be a non-empty list of images."
        # Check if all images in ims_moving have the same shape
        shape = ims_moving[0].shape
        for im in ims_moving:
            assert im.shape == shape, "All images in ims_moving must have the same shape."
        # Check if template_method is valid
        valid_template_methods = {'sequential', 'image'}
        assert template_method in valid_template_methods, f"template_method must be one of {valid_template_methods}"
        # Check if mode_transform is valid
        valid_mode_transforms = {'createOptFlow_DeepFlow', 'calcOpticalFlowFarneback'}
        assert mode_transform in valid_mode_transforms, f"mode_transform must be one of {valid_mode_transforms}"

        # Warn if any images have values below 0 or NaN
        found_0 = np.any([np.any(im < 0) for im in ims_moving])
        found_nan = np.any([np.any(np.isnan(im)) for im in ims_moving])
        warnings.warn(f"Found images with values below 0: {found_0}. Found images with NaN values: {found_nan}") if found_0 or found_nan else None

        H, W = ims_moving[0].shape[:2]
        self._HW = (H,W) if self._HW is None else self._HW
        x_grid, y_grid = np.meshgrid(np.arange(0., W).astype(np.float32), np.arange(0., H).astype(np.float32))

        ims_moving, template = self._fix_input_images(ims_moving=ims_moving, template=template, template_method=template_method)
        norm_factor = np.nanmax([np.nanmax(im) for im in ims_moving])
        template_norm   = np.array(template * (template > 0) * (1/norm_factor) * 255, dtype=np.uint8) if template_method == 'image' else None
        ims_moving_norm = [np.array(im * (im > 0) * (1/np.nanmax(im)) * 255, dtype=np.uint8) for im in ims_moving]

        print(f'Finding nonrigid registration warps with mode: {mode_transform}, template_method: {template_method}') if self._verbose else None
        remappingIdx_raw = []
        for ii, im_moving in tqdm(enumerate(ims_moving_norm), desc='Finding nonrigid registration warps', total=len(ims_moving_norm), unit='image', disable=not self._verbose):
            if template_method == 'sequential':
                ## warp images before template forward (t1->t2->t3->t4)
                if ii < template:
                    im_template = ims_moving_norm[ii+1]
                ## warp template to itself
                elif ii == template:
                    im_template = ims_moving_norm[ii]
                ## warp images after template backward (t4->t3->t2->t1)
                elif ii > template:
                    im_template = ims_moving_norm[ii-1]
            elif template_method == 'image':
                im_template = template_norm

            if im_template.ndim == 3:
                im_template = im_template.mean(2)
            if im_moving.ndim == 3:
                im_moving = im_moving.mean(2)

            if mode_transform == 'calcOpticalFlowFarneback':
                self._kwargs_method = {
                    'pyr_scale': 0.3, 
                    'levels': 3,
                    'winsize': 128, 
                    'iterations': 7,
                    'poly_n': 7, 
                    'poly_sigma': 1.5,
                    'flags': cv2.OPTFLOW_FARNEBACK_GAUSSIAN, ## = 256
                } if kwargs_mode_transform is None else kwargs_mode_transform
                flow_tmp = cv2.calcOpticalFlowFarneback(
                    prev=im_template,
                    next=im_moving, 
                    flow=None, 
                    **self._kwargs_method,
                )
        
            elif mode_transform == 'createOptFlow_DeepFlow':
                flow_tmp = cv2.optflow.createOptFlow_DeepFlow().calc(
                    im_template,
                    im_moving,
                    None
                )

            remappingIdx_raw.append(flow_tmp + np.stack([x_grid, y_grid], axis=-1))

        # compose warp transforms
        print('Composing nonrigid warp matrices...') if self._verbose else None
        
        self.remappingIdx_nonrigid = []
        if template_method == 'sequential':
            ## compose warps before template forward (t1->t2->t3->t4)
            for ii in np.arange(0, template):
                warp_composed = self._compose_warps(
                    warp_0=remappingIdx_raw[ii], 
                    warps_to_add=remappingIdx_raw[ii+1:template+1],
                    warpMat_or_remapIdx='remapIdx',
                )
                self.remappingIdx_nonrigid.append(warp_composed)
            ## compose template to itself
            self.remappingIdx_nonrigid.append(remappingIdx_raw[template])
            ## compose warps after template backward (t4->t3->t2->t1)
            for ii in np.arange(template+1, len(ims_moving)):
                warp_composed = self._compose_warps(
                    warp_0=remappingIdx_raw[ii], 
                    warps_to_add=remappingIdx_raw[template:ii][::-1],
                    warpMat_or_remapIdx='remapIdx',
                    )
                self.remappingIdx_nonrigid.append(warp_composed)
        ## no composition when template_method == 'image'
        elif template_method == 'image':
            self.remappingIdx_nonrigid = remappingIdx_raw

        if remappingIdx_init is not None:
            self.remappingIdx_nonrigid = [self._compose_warps(warp_0=remappingIdx_init[ii], warps_to_add=[warp], warpMat_or_remapIdx='remapIdx') for ii, warp in enumerate(self.remappingIdx_nonrigid)]

        self.remappingIdx_nonrigid = np.stack(self.remappingIdx_nonrigid, axis=0)

        return self.remappingIdx_nonrigid
        
    def transform_images_geometric(
        self, 
        ims_moving: np.ndarray, 
        remappingIdx: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Transforms images based on geometric registration warps.

        Args:
            ims_moving (np.ndarray): 
                The images to be transformed. *(N, H, W)*
            remappingIdx (Optional[np.ndarray]): 
                An array specifying how to remap the images. If ``None``, the
                remapping index from geometric registration is used. (Default is
                ``None``)

        Returns:
            (np.ndarray): 
                ims_registered_geo (np.ndarray):
                    The images after applying the geometric registration warps.
                    *(N, H, W)*
        """
        remappingIdx = self.remappingIdx_geo if remappingIdx is None else remappingIdx
        print('Applying geometric registration warps to images...') if self._verbose else None
        self.ims_registered_geo = self.transform_images(ims_moving=ims_moving, remappingIdx=remappingIdx)
        return self.ims_registered_geo
    
    def transform_images_nonrigid(
        self, 
        ims_moving: np.ndarray, 
        remappingIdx: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Transforms images based on non-rigid registration warps.

        Args:
            ims_moving (np.ndarray): 
                The images to be transformed. *(N, H, W)*
            remappingIdx (Optional[np.ndarray]): 
                An array specifying how to remap the images. If ``None``, the
                remapping index from non-rigid registration is used. (Default is
                ``None``)

        Returns:
            (np.ndarray): 
                ims_registered_nonrigid (np.ndarray): 
                    The images after applying the non-rigid registration warps.
                    *(N, H, W)*
        """
        remappingIdx = self.remappingIdx_nonrigid if remappingIdx is None else remappingIdx
        print('Applying nonrigid registration warps to images...') if self._verbose else None
        self.ims_registered_nonrigid = self.transform_images(ims_moving=ims_moving, remappingIdx=remappingIdx)
        return self.ims_registered_nonrigid
    
    def transform_images(
        self, 
        ims_moving: List[np.ndarray], 
        remappingIdx: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Transforms images using the specified remapping index.

        Args:
            ims_moving (List[np.ndarray]): 
                The images to be transformed. List of arrays with shape: *(H,
                W)* or *(H, W, C)*
            remappingIdx (List[np.ndarray]): 
                The remapping index to apply to the images.

        Returns:
            (List[np.ndarray]): 
                ims_registered (List[np.ndarray]): 
                    The transformed images. *(N, H, W)*
        """

        ims_registered = []
        for ii, (im_moving, remapIdx) in enumerate(zip(ims_moving, remappingIdx)):
            remapper = functools.partial(
                helpers.remap_images,
                remappingIdx=remapIdx,
                backend='cv2',
                interpolation_method='linear',
                border_mode='constant',
                border_value=float(im_moving.mean()),
            )
            im_registered = np.stack([remapper(im_moving[:,:,ii]) for ii in range(im_moving.shape[2])], axis=-1) if im_moving.ndim==3 else remapper(im_moving)
            ims_registered.append(im_registered)
        return ims_registered
    

    def _compose_warps(
        self, 
        warp_0: np.ndarray, 
        warps_to_add: List[np.ndarray], 
        warpMat_or_remapIdx: str = 'remapIdx'
    ) -> np.ndarray:
        """
        Composes a series of warps into a single warp.
        RH 2023

        Args:
            warp_0 (np.ndarray): 
                The initial warp.
            warps_to_add (List[np.ndarray]): 
                A list of warps to add to the initial warp.
            warpMat_or_remapIdx (str): 
                Determines the function to use for composition. Can be either 
                'warpMat' or 'remapIdx'. (Default is 'remapIdx')

        Returns:
            (np.ndarray): 
                warp_out (np.ndarray): 
                    The resulting warp after composition.
        """
        if warpMat_or_remapIdx == 'warpMat':
            fn_compose = helpers.compose_transform_matrices
        elif warpMat_or_remapIdx == 'remapIdx':
            fn_compose = functools.partial(helpers.compose_remappingIdx, method='linear', fill_value=None, bounds_error=False)
        else:
            raise ValueError(f'warpMat_or_remapIdx must be one of ["warpMat", "remapIdx"]')
        
        if len(warps_to_add) == 0:
            return warp_0
        else:
            warp_out = warp_0.copy()
            for warp_to_add in warps_to_add:
                warp_out = fn_compose(warp_out, warp_to_add)
            return warp_out

    def transform_points(
        self,
        points: np.ndarray,
        remappingIdx: np.ndarray,
    ):
        """
        Warps points through the supplied remapping index field. RH 2022

        Args:
            points (np.ndarray):
                Points to warp as ``(x, y)`` pairs. shape: *(n_points, 2)*,
                dtype: *float*.
            remappingIdx (np.ndarray):
                Remapping index field that maps output ``(x, y)`` coordinates
                to source ``(x, y)`` coordinates. shape: *(H, W, 2)*, dtype:
                *float*. Last dim is ``(x, y)``.

        Returns:
            (np.ndarray):
                points_remap (np.ndarray):
                    Warped points clipped to image bounds. shape:
                    *(n_points, 2)*.
        """
        points_remap = helpers.remap_points(
            points=points,
            remappingIdx=remappingIdx,
            interpolation='linear',
            fill_value=None,
        )

        ## Clip points to image size
        points_remap[:, 0] = np.clip(points_remap[:, 0], 0, remappingIdx.shape[1] - 1)
        points_remap[:, 1] = np.clip(points_remap[:, 1], 0, remappingIdx.shape[0] - 1)

        return points_remap


    def get_flowFields(
        self, 
        remappingIdx: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Returns the flow fields based on the remapping indices.

        Args:
            remappingIdx (Optional[np.ndarray]): 
                The indices for remapping the flow fields. 
                If ``None``, geometric or nonrigid registration must be performed first. 
                (Default is ``None``)

        Returns:
            (List[np.ndarray]): 
                flow_fields (List[np.ndarray]): 
                    The transformed flow fields.
        """
        if remappingIdx is None:
            assert (self.remappingIdx_geo is not None) or (self.remappingIdx_nonrigid is not None), 'If remappingIdx is not provided, then geometric or nonrigid registration must be performed first.'
            remappingIdx = self.remappingIdx_nonrigid if self.remappingIdx_nonrigid is not None else self.remappingIdx_geo
        return [helpers.remappingIdx_to_flowField(remap) for remap in remappingIdx]
    
    def _fix_input_images(
        self,
        ims_moving: List[np.ndarray],
        template: Union[int, np.ndarray],
        template_method: str,
    ) -> Tuple[int, List[np.ndarray]]:
        """
        Converts the input images and template to float32 dtype if they are not
        already. 

        Warnings are printed for any conversions made. The method for selecting
        the template image can either be **'image'** or **'sequential'**.

        Args:
            ims_moving (List[np.ndarray]):
                A list of input images. Images should be of type ``np.float32``,
                if not, they will be converted to it.
            template (Union[int, np.ndarray]): 
                The index or actual template image. Depending on the
                `template_method`, this could be an integer index, a float
                representing a fractional index, or a numpy array representing
                the actual template image.
            template_method (str):
                Method for selecting the template image. Either \n
                * ``'image'``: template is considered as an image
                 (``np.ndarray``) or as an index (``int`` or ``float``)
                 referring to the list of images (``ims_moving``). 
                * ``'sequential'``: template is considered as a sequential index
                  (``int`` or ``float``) referring to the list of images
                  (``ims_moving``). \n

        Returns:
            (Tuple[int, List[np.ndarray]]): tuple containing:
                template (int):
                    Index of the template in the list of images.
                ims_moving (List[np.ndarray]):
                    List of converted images.

        Example:
            .. highlight:: python
            .. code-block:: python

                ims_moving, template = _fix_input_images(ims_moving, template,
                'image')
        """
        ## convert images to float32 and warn if they are not
        print(f'WARNING: ims_moving are not all dtype: np.float32, found {np.unique([im.dtype for im in ims_moving])}, converting...') if any(im.dtype != np.float32 for im in ims_moving) else None
        ims_moving = [im.astype(np.float32) for im in ims_moving]    

        if template_method == 'image':
            if isinstance(template, int):
                assert 0 <= template < len(ims_moving), f'template must be between 0 and {len(ims_moving)-1}, not {template}'
                print(f'WARNING: template image is not dtype: np.float32, found {ims_moving[template].dtype}, converting...') if ims_moving[template].dtype != np.float32 else None
                template = ims_moving[template]
            elif isinstance(template, float):
                assert 0.0 <= template <= 1.0, f'template must be between 0.0 and 1.0, not {template}'
                idx = int(len(ims_moving) * template)
                print(f'Converting float fractional index to integer index: {template} -> {idx}')
                template = ims_moving[idx] # take the image at the specified fractional index
            elif isinstance(template, np.ndarray):
                assert template.ndim in [2, 3], f'template must be 2D or 3D, not {template.ndim}'
            else:
                raise ValueError(f'template must be np.ndarray or int or float between 0.0-1.0, not {type(template)}')
            if template.dtype != np.float32:
                print(f'WARNING: template image is not dtype: np.float32, found {template.dtype}, converting...')
                template = template.astype(np.float32)        

        elif template_method == 'sequential':
            assert isinstance(template, (int, float)), f'template must be int or float between 0.0-1.0, not {type(template)}'
            if isinstance(template, float):
                assert 0.0 <= template <= 1.0, f'template must be between 0.0 and 1.0, not {template}'
                idx = int(len(ims_moving) * template)
                print(f'Converting float fractional index to integer index: {template} -> {idx}')
                template = idx
            assert 0 <= template < len(ims_moving), f'template must be between 0 and {len(ims_moving)-1}, not {template}'

        return ims_moving, template

def clahe(
    im: np.ndarray, 
    grid_size: int = 50, 
    clipLimit: int = 0, 
    normalize: bool = True,
) -> np.ndarray:
    """
    Perform Contrast Limited Adaptive Histogram Equalization (CLAHE) on an image.

    Args:
        im (np.ndarray):
            Input image.
        grid_size (int):
            Size of the grid. See ``cv2.createCLAHE`` for more info. 
            (Default is *50*)
        clipLimit (int):
            Clip limit. See ``cv2.createCLAHE`` for more info. 
            (Default is *0*)
        normalize (bool):
            Whether to normalize the output image. 
            (Default is ``True``)
        
    Returns:
        (np.ndarray): 
            im_out (np.ndarray): 
                Output image after applying CLAHE.
    """
    import cv2
    im_tu = (im / im.max())*(2**8) if normalize else im
    im_tu = (im_tu/10).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid_size, grid_size))
    if im.ndim == 2:
        im_c = clahe.apply(im_tu.astype(np.uint16))
    elif im.ndim == 3:
        im_c = np.stack([clahe.apply(im_tu[:,:,i].astype(np.uint16)) for i in range(im.shape[2])], axis=2)
    return im_c
