"""HDF5 utilities: hierarchical traversal, group I/O, and bulk-close helpers.

Convenience wrappers around :mod:`h5py` for the face-rhythm project. Nothing
here is CUDA- or video-specific; the module is safe to import anywhere.
"""

import gc

import h5py
import numpy as np

def close_all_h5():
    """
    Closes every open :class:`h5py.File` object found in the Python workspace.

    Iterates over all live objects via :mod:`gc` and calls ``close`` on any
    :class:`h5py.File` instance. Falls back to
    ``tables.file._open_files.close_all`` if the primary loop raises. Adapted
    from https://stackoverflow.com/questions/29863342/close-an-open-h5py-data-file.
    """
    try:
        for obj in gc.get_objects():   # Browse through ALL objects
            if isinstance(obj, h5py.File):   # Just HDF5 files
                try:
                    obj.close()
                except (OSError, ValueError, RuntimeError):
                    pass # Was already closed
    except Exception as e:
        print(f"Error closing h5 files. Will try again using `tables._open_files.close_all()`. Error: {e}")
        import tables
        tables.file._open_files.close_all()
    
    gc.collect()



def show_group_items(hObj):
    """
    Prints the items at the top hierarchical level of an HDF5 object or dict. RH 2021

    See :func:`show_item_tree` for a full recursive listing.

    Args:
        hObj (object):
            Hierarchical object: an :class:`h5py.File`, :class:`h5py.Group`, or a
            Python ``dict``.

    Example:
        .. highlight:: python
        .. code-block:: python

            with h5py.File(path, 'r') as f:
                h5_handling.show_group_items(f)
    """

    for ii,val in enumerate(list(iter(hObj))):
        if isinstance(hObj[val] , h5py.Group) or isinstance(hObj[val]):
            print(f'{ii+1}. {val}:----------------')
        if isinstance(hObj[val] , dict):
            print(f'{ii+1}. {val}:----------------')
        else:
            if hasattr(hObj[val] , 'shape') and hasattr(hObj[val] , 'dtype'):
                print(f'{ii+1}. {val}:   shape={hObj[val].shape} , dtype={hObj[val].dtype}')
            else:
                print(f'{ii+1}. {val}:   type={type(hObj[val])}')



def show_item_tree(hObj=None , path=None, depth=None, show_metadata=True, print_metadata=False, indent_level=0):
    """
    Recursively prints the items and groups in an HDF5 object or dict. RH 2021

    Args:
        hObj (object):
            Hierarchical object: an :class:`h5py.File`, :class:`h5py.Group`, or a
            Python ``dict``. Ignored when ``path`` is provided. (Default is
            ``None``)
        path (Optional[object]):
            Path-like to an HDF5 file to open in read mode. If not ``None``,
            the file is opened and traversed in place of ``hObj``. (Default is
            ``None``)
        depth (Optional[int]):
            Maximum number of hierarchical levels to descend. ``None`` means
            unlimited. (Default is ``None``)
        show_metadata (bool):
            If ``True``, list per-node metadata attributes alongside items.
            (Default is ``True``)
        print_metadata (bool):
            If ``True``, also print the value of each metadata attribute;
            otherwise only its shape and dtype are shown. (Default is ``False``)
        indent_level (int):
            Internal recursion bookkeeping for indentation; users should leave
            this at the default. (Default is ``0``)

    Example:
        .. highlight:: python
        .. code-block:: python

            with h5py.File(path, 'r') as f:
                h5_handling.show_item_tree(f)
    """

    if depth is None:
        depth = int(10000000000000000000)
    else:
        depth = int(depth)

    if depth < 0:
        return

    if path is not None:
        with h5py.File(path , 'r') as f:
            show_item_tree(hObj=f, path=None, depth=depth-1, show_metadata=show_metadata, print_metadata=print_metadata, indent_level=indent_level)
    else:
        indent = f'  '*indent_level
        if hasattr(hObj, 'attrs') and show_metadata:
            for ii,val in enumerate(list(hObj.attrs.keys()) ):
                if print_metadata:
                    print(f'{indent}METADATA: {val}: {hObj.attrs[val]}')
                else:
                    print(f'{indent}METADATA: {val}: shape={hObj.attrs[val].shape} , dtype={hObj.attrs[val].dtype}')
        
        for ii,val in enumerate(list(iter(hObj))):
            if isinstance(hObj[val], h5py.Group):
                print(f'{indent}{ii+1}. {val}:----------------')
                show_item_tree(hObj[val], depth=depth-1, show_metadata=show_metadata, print_metadata=print_metadata , indent_level=indent_level+1)
            elif isinstance(hObj[val], dict):
                print(f'{indent}{ii+1}. {val}:----------------')
                show_item_tree(hObj[val], depth=depth-1, show_metadata=show_metadata, print_metadata=print_metadata , indent_level=indent_level+1)
            else:
                if hasattr(hObj[val], 'shape') and hasattr(hObj[val], 'dtype'):
                    print(f'{indent}{ii+1}. {val}:    '.ljust(20) + f'shape={hObj[val].shape} ,'.ljust(20) + f'dtype={hObj[val].dtype}')
                else:
                    print(f'{indent}{ii+1}. {val}:    '.ljust(20) + f'type={type(hObj[val])}')


def make_h5_tree(dict_obj , h5_obj , group_string='', use_compression=False, track_order=True):
    """
    Recursively writes a Python dict into an HDF5 group/dataset tree. RH 2021

    Intended to be called by :func:`write_dict_to_h5`; using it directly is
    **not** recommended.

    Args:
        dict_obj (dict):
            Source dictionary whose hierarchy and leaf values become groups and
            datasets, respectively.
        h5_obj (h5py.File):
            Open HDF5 file (or group) into which the tree is written.
        group_string (str):
            Path of the current HDF5 group within ``h5_obj`` during recursion.
            An empty string is treated as the root ``'/'``. (Default is ``''``)
        use_compression (bool):
            If ``True``, write each dataset with gzip level 9 compression.
            (Default is ``False``)
        track_order (bool):
            If ``True``, set :func:`h5py.get_config` to preserve insertion
            order of items. (Default is ``True``)
    """
    ## Set track_order to True to keep track of the order of the items in the dict
    ##  This is useful for reading the dict back in from the h5 file
    h5py.get_config().track_order = track_order
    for ii,(key,val) in enumerate(dict_obj.items()):
        if group_string=='':
            group_string='/'
        if isinstance(val , dict):
            # print(f'making group:  {key}')
            h5_obj[group_string].create_group(key)
            make_h5_tree(val , h5_obj[group_string] , f'{group_string}/{key}', use_compression=use_compression)
        else:
            ## cast to 'S' type if string so that it doesn't become '|O' object type in h5 file
            if isinstance(val, str):
                val = np.array(val, dtype=np.bytes_)
            
            # print(f'saving:  {group_string}: {key}')
            kwargs_compression = {'compression': 'gzip', 'compression_opts': 9} if use_compression else {}
            h5_obj[group_string].create_dataset(key , data=val, **kwargs_compression)
def write_dict_to_h5(
    path_save, 
    input_dict, 
    use_compression=False, 
    track_order=True, 
    write_mode='w-', 
    show_item_tree_pref=True
):
    """
    Writes a Python dict to an HDF5 file, mirroring its hierarchy and data. RH 2021

    Wraps :func:`make_h5_tree` and optionally prints the resulting tree.

    Args:
        path_save (object):
            Full path of the file to write. ``str`` or :class:`pathlib.Path`.
        input_dict (dict):
            Dictionary whose leaves are HDF5-writable values (typically
            :class:`numpy.ndarray` or strings).
        use_compression (bool):
            If ``True``, write each dataset with gzip compression. (Default is
            ``False``)
        track_order (bool):
            If ``True``, preserve dict insertion order in the HDF5 file.
            (Default is ``True``)
        write_mode (str):
            File-open mode forwarded to :class:`h5py.File`. Either \n
            * ``'w'``: Overwrite any existing file.
            * ``'w-'``: Refuse to overwrite an existing file. \n
            (Default is ``'w-'``)
        show_item_tree_pref (bool):
            If ``True``, print the resulting HDF5 hierarchy after writing.
            (Default is ``True``)
    """
    with h5py.File(path_save , write_mode) as hf:
        make_h5_tree(input_dict , hf , '', use_compression=use_compression, track_order=track_order)
        if show_item_tree_pref:
            print(f'==== Successfully wrote h5 file. Displaying h5 hierarchy ====')
            show_item_tree(hf)


def simple_load(filepath, return_dict=True, verbose=False):
    """
    Loads an HDF5 file and returns it as a nested ``dict`` or an open file. RH 2023

    Args:
        filepath (object):
            Full path of the file to read. ``str`` or :class:`pathlib.Path`.
        return_dict (bool):
            If ``True``, return a nested ``dict`` whose keys are group names
            and whose leaves are the dataset arrays. If ``False``, return the
            open :class:`h5py.File` object instead. (Default is ``True``)
        verbose (bool):
            If ``True``, print the file's hierarchy via :func:`show_item_tree`
            before returning. (Default is ``False``)

    Returns:
        (object):
            data (object):
                Either a nested ``dict`` of arrays (when ``return_dict`` is
                ``True``) or an open :class:`h5py.File` handle.
    """
    if return_dict:
        with h5py.File(filepath, 'r') as h5_file:
            if verbose:
                print(f'==== Loading h5 file with hierarchy: ====')
                show_item_tree(h5_file)
            result = {}
            def visitor_func(name, node):
                # Split name by '/' and reduce to nested dict
                keys = name.split('/')
                sub_dict = result
                for key in keys[:-1]:
                    sub_dict = sub_dict.setdefault(key, {})

                if isinstance(node, h5py.Dataset):
                    sub_dict[keys[-1]] = node[...]
                elif isinstance(node, h5py.Group):
                    sub_dict.setdefault(keys[-1], {})

            h5_file.visititems(visitor_func)            
            return result
    else:
        return h5py.File(filepath, 'r')

def h5Obj_to_dict(hObj):
    """
    Converts an :mod:`h5py` group or file into a nested Python ``dict``. RH 2023

    Args:
        hObj (object):
            An :class:`h5py.File` or :class:`h5py.Group` to traverse.

    Returns:
        (dict):
            h5_dict (dict):
                Nested dictionary mirroring the HDF5 hierarchy. Datasets are
                materialized via ``[()]``.
    """
    h5_dict = {}
    for ii,val in enumerate(list(iter(hObj))):
        if isinstance(hObj[val], h5py.Group):
            h5_dict[val] = h5Obj_to_dict(hObj[val])
        else:
            h5_dict[val] = hObj[val][()]
    return h5_dict


def simple_save(
    dict_to_save, 
    path=None, 
    use_compression=False, 
    track_order=True, 
    write_mode='w-', 
    verbose=False
):
    """
    Saves a Python dict to an HDF5 file or appends it to an existing one. RH 2021

    Args:
        dict_to_save (dict):
            Dictionary to save to the HDF5 file.
        path (object):
            Full path of the file to write. ``str`` or :class:`pathlib.Path`.
            (Default is ``None``)
        use_compression (bool):
            If ``True``, write each dataset with gzip compression. (Default is
            ``False``)
        track_order (bool):
            If ``True``, preserve dict insertion order in the HDF5 file.
            (Default is ``True``)
        write_mode (str):
            File-open mode forwarded to :class:`h5py.File`. Either \n
            * ``'w'``: Overwrite any existing file.
            * ``'w-'``: Refuse to overwrite an existing file.
            * ``'a'``: Append a new dataset to an existing file. \n
            (Default is ``'w-'``)
        verbose (bool):
            If ``True``, print the resulting HDF5 hierarchy after writing.
            (Default is ``False``)
    """

    write_dict_to_h5(
        path, 
        dict_to_save, 
        use_compression=use_compression, 
        track_order=track_order,
        write_mode=write_mode, 
        show_item_tree_pref=verbose
    )


def merge_helper(d, group):
    """
    Recursively merges a dictionary into an open :class:`h5py.Group`.

    Sub-dictionaries map to subgroups; non-dict values are written as
    datasets, replacing any existing dataset with the same name.

    Args:
        d (dict):
            Dictionary containing the data to merge.
        group (object):
            Target :class:`h5py.Group` (or :class:`h5py.File`) to merge into.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            # If the value is a dictionary, check if the group already exists, and either skip it or merge the data
            if key in group:
                merge_helper(value, group[key])
            else:
                subgroup = group.create_group(key)
                merge_helper(value, subgroup)
        else:
            # If the value is not a dictionary, convert it to a numpy array and create a dataset
            if key in group:
                del group[key]
            group.create_dataset(key, data=value)
def merge_dict_into_h5_file(d, filepath=None, h5Obj=None,):
    """
    Merges a dictionary into an existing HDF5 file or open file object.

    Wraps :func:`merge_helper`, which recursively walks the dict and merges
    each level into the matching HDF5 group. Exactly one of ``filepath`` or
    ``h5Obj`` must be supplied.

    Args:
        d (dict):
            Dictionary containing the data to merge.
        filepath (Optional[str]):
            Path to an HDF5 file to open in append mode. Do not specify when
            ``h5Obj`` is provided. (Default is ``None``)
        h5Obj (object):
            Open :class:`h5py.File` object to merge into. Do not specify when
            ``filepath`` is provided. (Default is ``None``)
    """
    if filepath is None and h5Obj is None:
        raise ValueError('Either filepath or h5Obj must be specified.')
    elif filepath is not None and h5Obj is not None:
        raise ValueError('Only one of filepath or h5Obj must be specified.')
    
    elif filepath is not None:
        with h5py.File(filepath, 'a') as file:
            merge_helper(d, file)
    elif h5Obj is not None:
        merge_helper(d, h5Obj)
