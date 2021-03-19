Data Organization
=================
Here I'll both define the terminology we use for talking about data, and
I'll also give some recommendations on how to structure your data when using Face Rhythm.

Terminology
-----------
Session: A session corresponds to a specific continuous experiment. This experiment was recorded and then saved as either
one single video or a series of chunked videos. If you've saved your session as multiple videos, we assume that the videos are
continuous. In other words, we assume that the videos can be ordered and that the last frame of one video is immediately followed
by the first frame of the next. This is important for when we run tracking of points from frame to frame.

Trial: Within a session, you might have multiple trials when the experimental subject is asked to perform different tasks.
If you have a trial-like structure, we assume that you provide a list of lists of indices where each list of indices
specifies the frames of the session that correspond to the trial. If your session is chunked, the trial indices should be absolute
(aka correspond to the frame number wrt to all the frames in the entire session).

Organization
------------

1. Basic
This method of data organization works great if you're just trying analyze a single video or a set of videos from a single session.
We just assume that you provide us a folder with videos in it and a file name stem that helps us narrow down what videos you want.
The file name stem is useful if there are other videos in this folder that you want to ignore.

Example

Settings:
video_folder = path/to/session1
file_prefix = 'session1_chunk'

Organization:
session1
- session1_chunk1.avi <-analyzed
- session1_chunk2.avi <-analyzed
- session1_test.avi <-ignored


2. Trials
You may have divided up your session into trials. As described above, we just expect that you provide a list of list
of frame indices indicating which frames belong to which trial. For example, if the total video is 6 frames long with
two trials of length three, the indices would look like [[0,1,2],[3,4,5]]. To add trials, just add a trial index file.

Example

Settings:
trials = True
video_folder = path/to/session1
file_prefix = 'session1_chunk'

Organization:
session1
- session1_chunk1.avi <-analyzed
- session1_chunk2.avi <-analyzed
- session1_test.avi <-ignored
- trial_indices.npy


3. Multiple Sessions
You may want to analyze multiple sessions within a single Face Rhythm analysis. This is especially useful if you've
optimized your workflow, and you want to start pushing more data through this pipeline. For general sanity, we
require a somewhat strict organization of sessions to keep the workflow simple.

We expect that you have a top level data folder full of session folders. We then expect that
each session be a single folder with one or more videos inside it. To accommodate this setup, our parameters
are slightly different. Now we list the data folder and then set a session prefix for which sessions we want.

Example

Settings:
data_folder = path/to/all_data
session_prefix = 'session'

Organization:
all_data
- session1
   - session1_chunk1.avi <-analyzed
- session2
   - session2_chunk1.avi <- analyzed



