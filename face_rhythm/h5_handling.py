###########################
####### H5_HANDLING #######
###########################

import gc

import h5py
import numpy as np

def close_all_h5():
    '''
    Closes all h5 objects in workspace. Not tested thoroughly.
    from here: https://stackoverflow.com/questions/29863342/close-an-open-h5py-data-file
    '''
    try:
        for obj in gc.get_objects():   # Browse through ALL objects
            if isinstance(obj, h5py.File):   # Just HDF5 files
                try:
                    obj.close()
                except:
                    pass # Was already closed
    except Exception as e:
        print(f"Error closing h5 files. Will try again using `tables._open_files.close_all()`. Error: {e}")
        import tables
        tables.file._open_files.close_all()
    
    gc.collect()



def show_group_items(hObj):
    '''
    Simple function that displays all the items and groups in the
     highest hierarchical level of an h5 object or python dict.
    See 'show_item_tree' to view the whole tree
    RH 2021

    Args:
        hObj: 'hierarchical Object' hdf5 object or subgroup object OR python dictionary
    
    ##############

    example usage:
        with h5py.File(path , 'r') as f:
            h5_handling.show_group_items(f)
    '''

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
    '''
    Recursive function that displays all the items 
     and groups in an h5 object or python dict
    RH 2021

    Args:
        hObj:
            'hierarchical Object'. hdf5 object OR python dictionary
        path (Path or string):
            If not None, then path to h5 object is used instead of hObj
        depth (int):
            how many levels deep to show the tree
        show_metadata (bool): 
            whether or not to list metadata with items
        print_metadata (bool): 
            whether or not to show values of metadata items
        indent_level: 
            used internally to function. User should leave blank

    ##############
    
    example usage:
        with h5py.File(path , 'r') as f:
            h5_handling.show_item_tree(f)
    '''

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
    '''
    This function is meant to be called by write_dict_to_h5. It probably shouldn't be called alone.
    This function creates an h5 group and dataset tree structure based on the hierarchy and values within a python dict.
    There is a recursion in this function.
    RH 2021
    '''
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
                val = np.array(val, dtype=np.string_)
            
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
    '''
    Writes an h5 file that matches the hierarchy and data within a python dict.
    This function calls the function 'make_h5_tree'
    RH 2021
   
    Args:
        path_save (string or Path): 
            Full path name of file to write
        input_dict (dict): 
            Dict containing only variables that can be written as a 'dataset' in an h5 file (generally np.ndarrays and strings)
        use_compression (bool):
            Whether or not to use compression when writing the h5 file
        track_order (bool):
            Whether or not to keep track of the order of the items in the dict
        write_mode ('w' or 'w-'): 
            The priveleges of the h5 file object. 'w' will overwrite. 'w-' will not overwrite
        show_item_tree_pref (bool): 
            Whether you'd like to print the item tree or not
    '''
    with h5py.File(path_save , write_mode) as hf:
        make_h5_tree(input_dict , hf , '', use_compression=use_compression, track_order=track_order)
        if show_item_tree_pref:
            print(f'==== Successfully wrote h5 file. Displaying h5 hierarchy ====')
            show_item_tree(hf)


def simple_load(filepath, return_dict=True, verbose=False):
    """
    Returns a dictionary object containing the groups
    as keys and the datasets as values from
    given hdf file.
    RH 2023

    Args:
        filepath (string or Path): 
            Full path name of file to read.
        return_dict (bool):
            Whether or not to return a dict object (True)
            or an h5py object (False)
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
    '''
    Converts an h5py object to a python dict object
    RH 2023
    '''
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
    Saves a python dict to an hdf file.
    Also allows for adding new data to
     an existing hdf file.
    RH 2021

    Args:
        dict_to_save (dict):
            Python dict to save to hdf file.
        path (string or Path):
            Full path name of file to write.
        write_mode ('w', 'w-', 'a'):
            The priveleges of the h5 file object.
            'w' will overwrite.
            'w-' will not overwrite.
            'a' will append/add a new dataset to the h5 file.
        use_compression (bool):
            Whether or not to use compression when writing the h5 file
        track_order (bool):
            Whether or not to keep track of the order of the items in the dict
        verbose (bool):
            Whether or not to print out the h5 file hierarchy.
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
    Merge a dictionary into an existing HDF5 file identified
     by an h5py.File object.

    Args:
        d (dict): 
            The dictionary containing the data to be merged.
        group (h5py.Group): 
            The HDF5 group to which the data should be merged.
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
    Merge a dictionary into an existing HDF5 file identified
     by a file path.
    This function wraps a recursive function that goes through
     each hierarchical level of the input dictionary and merges
     it into the appropriate HDF5 group.

    Args:
        d (dict): 
            The dictionary containing the data to be merged.
        filepath (str): 
            The file path of the HDF5 file.
            Do not specify if hObj is specified.
        h5Obj (h5py.File):
            An h5py.File object.
            Do not specify if filepath is specified.
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
