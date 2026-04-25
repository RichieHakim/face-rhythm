Data Standards
==============

This page describes minimum requirements and recommendations for video data
used with face-rhythm.

----

Minimum requirements
--------------------

1. Codec is readable by FFMpeg:
   https://www.ffmpeg.org/general.html#Supported-File-Formats_002c-Codecs-or-Features

   a. Almost all codecs are readable, so just give it a go.

----

Video codec and compression
---------------------------

.. warning::

   Avoid video codecs that use keyframes or Periodic Intra Refreshes
   (most commonly h.264, h.265, x264). **We recommend using simpler
   codecs like MPEG-2 and MPEG-4, and minimizing compression.** x264 and
   similar codecs will work in the pipeline, but the compression method
   changes the temporal structure of points in space. In addition,
   separating the files into 'chunks' of trials results in loss of data
   and may cause errors.

**File conventions:**

a. Videos can be separated by trials (eg. ``session_4_trial_0001.avi``,
   ``session_4_trial_0002.avi``).

b. Videos can be 'chunked' up into parts (eg. ``session_4_0001.avi``,
   ``session_4_0002.avi``). This assumes that subsequent frames between
   files are continuous without any skips/jumps.

----

Frame rate
----------

.. note::

   Make sure the frame rate is more than twice the sample rate of any
   features of interest. For example: Mouse whisking is ~12 Hz, so if
   you want to extract that feature, the frame rate must be >24 Hz.

For mouse face videos, we recommend a frame rate of at least 60 Hz,
**ideally >90 Hz.**

----

Exposure time
-------------

Minimize motion blur: If the image is blurred, a significant amount of
critical information is lost. The core optic flow algorithm (`see OpenCV's
Lucas-Kanade tutorial
<https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html>`__)
depends on using high spatial frequency information to solve the optic flow
equations.

For mouse face videos, we recommend an exposure time of at most 5 ms,
**ideally <1 ms.**

----

Video continuity and focus
--------------------------

**In focus:** If the image is blurred, critical information is lost. The
optic flow algorithm depends on high spatial frequency information (see
Exposure time section above for the algorithmic reason).

**No 'jumps' in video:** Because the algorithm looks at changes between
frames, 'jumps' in the video can cause it to be unstable. This includes
black frames, significant frame drop events, etc.

----

Occlusions
----------

Face-rhythm was designed with mouse face video in mind, but many other
kinds of data can work as well. Below are some of the constraints.

**Minimal occlusions over region of interest:** The algorithm just tracks
points, so if an object sweeps across the field of view, the points will
move with that object. We have included some methods of cleaning up
moderate amounts of occlusion events by temporarily stopping point tracking
during occlusion events, but it is still best if the events can be
minimized.

For mouse face videos, grooming events are occlusive of the face, so it is
best to position the camera to minimize the observed occlusion by
forelimbs. Whiskers also occlude parts of the face, so parameters should
be tuned to minimize tracking of the occlusive whisker movements (or
maximize them if they are of interest!).

**Motion and positioning:**

- Feature positions should vary around a center. In other words, movement
  should not have very slow offsets.

- Features should not move significantly from its center position. In other
  words, movements should be relatively small.

----

Quality optimization checklist
-------------------------------

To improve tracking quality, consider the following in order of impact:

1.  Increase frame rate

2.  Increase resolution

3.  Decrease aperture (increase 'F-stop' of camera) to increase depth of
    field that is in focus

4.  Decrease exposure time per frame

5.  Get a brighter light (but don't over expose video) in order to achieve
    above recommendations

6.  Minimize large shadows (though increasing small shadows like over fur
    can increase local contrast, and thus point tracking quality)

7.  Minimize overexposed pixels

8.  Minimize compression of the video.

9.  Use a bit depth of at least 8 bits post-compression.

----

Hardware and acquisition
------------------------

If data streaming/saving is your bottleneck, and your camera is compatible
with Bonsai: https://bonsai-rx.org/, try it out. It might remove that
bottleneck. Open a GitHub issue if you would like reference code for
running FLIR cameras.
