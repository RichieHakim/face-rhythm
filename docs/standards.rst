Data Standards
============

Minimum video requirements
--------------------------
* Codec is readable by FFMpeg: https://www.ffmpeg.org/general.html#Supported-File-Formats_002c-Codecs-or-Features
* Almost all codecs are readable, so just give it a go.

Recommended video requirements
-----------------------------
* High frame rate: Make sure frame rate is more than twice the sample rate of any features of interest. For example: Mouse whisking is ~12 Hz, so if you want to extract that feature, the frame rate must be >24 Hz. For mouse face videos we recommend a frame rate of at least 60 Hz.
* In focus: The core optic flow algorithm (see here for details) depends on aligning the Fourier transforms of consecutive frames, so if the image is blurred, a significant amount of critical information is lost.
* Minimal motion blur: Similar reason to above.
* No ‘jumps’ in video: Because the algorithm looks at changes between frames, ‘jumps’ in the video can cause it to be unstable. This includes black frames, significant frame drop events, etc.

Recommended sample requirements
------------------------------
Face-rhythm was designed with mouse face video in mind, but many other kinds of data can work as well. Below are some of the constraints.

* Minimal occlusions over region of interest: The algorithm just tracks points, so if an object sweeps across the field of view, the points will move with that object. We have included some methods of cleaning up moderate amounts of occlusion events by temporarily stopping point tracking during occlusion events, but it is still best if the events can be minimized.
* For mouse face videos, grooming events are occlusive of the face, so it is best to position the camera to minimize the observed occlusion by forelimbs. Whiskers also occlude parts of the face, so parameters should be tuned to minimize tracking of the occlusive whisker movements (or maximize them if they are of interest!).
* Feature positions should vary around a center. In other words, movement should not have very slow offsets.
* Features should not move significantly from its center position.

Recommendations on how to optimize quality
---------------------------------------
* Increase frame rate
* Increase resolution
* Decrease aperture (increase ‘F-stop’ of camera) to increase depth of field that is in focus
* Decrease exposure time per frame
* Get a brighter light (but don’t over expose video)
* Minimize large shadows (though increasing small shadows like over fur, can increase local contrast, and thus point tracking quality)
* Minimize compression of the video. Avoid video codecs that use keyframes of Periodic Intra Refreshes (most commonly h.264, h.265, x264). We recommend using simpler codecs like MPEG-2 and MPEG-4, and minimizing compression.
* Use a bit depth of at least 8 bits post-compression.

