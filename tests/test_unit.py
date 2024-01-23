####### data_importing #######

## Dataset_videos
from face_rhythm.data_importing import Dataset_videos
class test_Dataset_videos:
    def __init__(self):
        pass

    ## Test if type checking works
    def test_type_checking(self):
        import pytest
        
        ## test if paths_videos is a list of strings or string
        conditions_error = [
            1,
            [1],
            [1, 2],
            ['test', 2],
            [1, 'test'],
        ]
        for condition in conditions_error:
            with pytest.raises(AssertionError):
                Dataset_videos(paths_videos=condition)

        ## test if paths_videos exist
        conditions_error = [
            ['test'],
            ['test', 'test'],
        ]
        for condition in conditions_error:
            with pytest.raises(AssertionError):
                Dataset_videos(paths_videos=condition)
