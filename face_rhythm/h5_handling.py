###########################
####### H5_HANDLING #######
###########################
## From BNPM

import gc
import h5py

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
    except:
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
    import hdfdict

    for ii,val in enumerate(list(iter(hObj))):
        if isinstance(hObj[val] , h5py.Group) or isinstance(hObj[val], hdfdict.hdfdict.LazyHdfDict):
            print(f'{ii+1}. {val}:----------------')
        if isinstance(hObj[val] , dict):
            print(f'{ii+1}. {val}:----------------')
        else:
            if hasattr(hObj[val] , 'shape') and hasattr(hObj[val] , 'dtype'):
                print(f'{ii+1}. {val}:   shape={hObj[val].shape} , dtype={hObj[val].dtype}')
            else:
                print(f'{ii+1}. {val}:   type={type(hObj[val])}')



def show_item_tree(hObj=None , path=None, show_metadata=True , print_metadata=False, indent_level=0):
    '''
    Recursive function that displays all the items 
     and groups in an h5 object or python dict
    RH 2021

    Args:
        hObj:
            'hierarchical Object'. hdf5 object OR python dictionary
        path (Path or string):
            If not None, then path to h5 object is used instead of hObj
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
    import hdfdict

    if path is not None:
        with h5py.File(path , 'r') as f:
            show_item_tree(hObj=f, path=None, show_metadata=show_metadata, print_metadata=print_metadata, indent_level=indent_level)
    else:
        indent = f'  '*indent_level
        if hasattr(hObj, 'attrs') and show_metadata:
            for ii,val in enumerate(list(hObj.attrs.keys()) ):
                if print_metadata:
                    print(f'{indent}METADATA: {val}: {hObj.attrs[val]}')
                else:
                    print(f'{indent}METADATA: {val}: shape={hObj.attrs[val].shape} , dtype={hObj.attrs[val].dtype}')
        
        for ii,val in enumerate(list(iter(hObj))):
            if isinstance(hObj[val] , h5py.Group) or isinstance(hObj[val], hdfdict.hdfdict.LazyHdfDict):
                print(f'{indent}{ii+1}. {val}:----------------')
                show_item_tree(hObj[val], show_metadata=show_metadata, print_metadata=print_metadata , indent_level=indent_level+1)
            elif isinstance(hObj[val] , dict):
                print(f'{indent}{ii+1}. {val}:----------------')
                show_item_tree(hObj[val], show_metadata=show_metadata, print_metadata=print_metadata , indent_level=indent_level+1)
            else:
                if hasattr(hObj[val] , 'shape') and hasattr(hObj[val] , 'dtype'):
                    print(f'{indent}{ii+1}. {val}:   shape={hObj[val].shape} , dtype={hObj[val].dtype}')
                else:
                    print(f'{indent}{ii+1}. {val}:   type={type(hObj[val])}')


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
            # print(f'saving:  {group_string}: {key}')
            if use_compression:
                h5_obj[group_string].create_dataset(key , data=val, compression='gzip', compression_opts=9)
            else:
                h5_obj[group_string].create_dataset(key , data=val)
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


def simple_load(path=None, verbose=False):
    """
    Returns a lazy dictionary object (specific
    to hdfdict package) containing the groups
    as keys and the datasets as values from
    given hdf file.
    RH 2021

    Args:
        path (string or Path): 
            Full path name of file to read.
        verbose (bool):
            Whether or not to print out the h5 file hierarchy.
    
    Returns:
        h5_dict (LazyHdfDict):
            LazyHdfDict object containing the groups
    """
    import hdfdict
    
    h5Obj = hdfdict.load(str(path), **{'mode': 'r'})
    
    if verbose:
        show_item_tree(hObj=h5Obj)
    
    return h5Obj


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