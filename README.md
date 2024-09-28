# Lightpaint

Lightpaint is a video processing script that takes an input video and outputs a video with frame-by-frame blending effects applied to the original video. The most common use case is lightpainting where a moving light source is significantly brighter than the dark scene, creating a trailing effect of the bright areas.

# Features

Lightpaint is for editing lightpainting videos to make light sources to leave a trail behind them which decay over time. There are different ways to achieve this effect and there's no "one-size-fits-all" way o doing this. However, the tool comes with some default values to get started; which may or may not work on your scenario. All the features can be customized via CLI parameters:

## Mode

Controls how bright pixels' brightness decay over time. Used with `--mode` (or `-m`) parameter. Default is `decay`.

- `decay`: The default mode. Brightness decays using a decay function (exponential or linear).
- `window`: Brightness for bright pixels will stay the same for `n` frames (where `n` can be specified by `--window_size` parameter) and then reset without a smooth decay. Please note that this mode is significantly slower than other modes and gets even slower with larger `--window_size` values.
- `cumulative`: All the frames are cumulatively added using the specified blend mode one after another, without decay or reset. So, in other words, for example in `lighten` or `screen` or `add` blend mode, if a pixel ever gets white, it will stay white for the rest of the video.

## Decay type

Controls whether the decaying of pixel brightness is exponential or linear when mode is `decay`. Used with `--decay_type` (or `-dt`) parameter. Default is `exponential`

- `exponential`: Pixel values will "decay" back to "darker" values (at least, in default brightening blend modes like `lighten` or `screen` etc.) using an exponential function.
- `linear` Pixel values will decay back using a linear value. Feels less natural though has its own style.

## Decay parameter

Controls the strength of the decay function. The higher this value is the faster the decay. Can be used with both linear and exponential. Used with `--decay_param` (or `-dp`) parameter. Default is `0.95`.

## Window size

Controls how many frames a bright pixel will stay bright until falling back to the current (darker) pixel value, when decay mode is `window`. Used with `--window_size` (or `-ws`). High values (e.g. 50+) will have a longer lasting effect for brightness effects but they will take _significantly_ longer to process.

## Blend mode

Controls the blend mode to apply to each frame over the previous frame. These are the standard blending modes that are used through computer graphics and design (e.g. Photoshop). Used with `--blend_mode` (or `-bm`) parameter. Default is `lighten`. While a variety of modes are provided as courtesy, `lighten` and sometimes `screen` generally work the best. Please experiment and shoutout to me if you find any other blend modes useful!

## Power

Controls the dropoff power for the blending effect between each frames. Can be used with any other mode and is also used to modulate some of the blending modes. Used with `--power` (or `-pw` parameter). Default is `1`.

## Bit depth

_WARNING: This parameter is experimental and may not work in many scenarios!_

Controls the bit depth of processing and output. Used with `--bit_depth` or `-bd` parameter. Default is `8` (SDR) and allowed values are `10`, `12`, `16`, and `32`, which they all act as HDR mode. In HDR mode, single frames are saved as output instead of a video file, _unless `--use_ffmpeg` flag is used_. Please note that ffmpeg HDR output works in very few scenarios and fails in most scenarios. 

## Use ffmpeg

Controls whether or not to use ffmpeg with HDR bit depths (higher than `8`). Experimental and works with very specific scenarios only. Used with `--use-ffmpeg` (or `-ff`),

# Getting started

Follow these steps to install and set up the necessary environment to run lightpaint_batch.py.

1. Install Python 3

Ensure that Python 3 is installed on your system. You can check the version by running the following command in your terminal (macOS/Linux) or command prompt (Windows):
```
python3 --version
```
(note: you can replace `python3` and `pip3` with `python` and `pip` if the default installation on your system is 3.x)

If Python 3 is not installed, download and install it from the official Python website, or package manager of your choice.


2. Install Required Python Packages

After installing Python, you need to install the required Python libraries to run the scripts. You can install them using pip (Python's package manager). Open a terminal or command prompt and run the following command:
```
pip install opencv-python numpy tqdm
```
This will install the following packages:
 - OpenCV (opencv-python): Used for video processing.
 - NumPy (numpy): Used for numerical operations and array handling.
 - TQDM (tqdm): Used for displaying progress bars during processing.


3. Download the Scripts

Download the lightpaint_batch.py and process_video.py scripts (or just clone this repo) and place them in a folder on your system. You can organize your project like this:
```
lightpainting/
├── lightpaint_batch.py
├── process_video.py
└── input_video.mp4  # (Your input video file)
```
(You don't need to name your video `input_video.mp4`, this is just an example)


4. Verify the Installation

To verify that everything is set up correctly, navigate to the folder where the scripts are located using your terminal or command prompt:

```
cd path/to/lightpainting
```
(replace `path/to/lightpainting` with the actual path in your system wherever you've put the scripts and the video)

Run the following command to verify that Python can find the necessary packages:
```
python3 -c "import cv2, numpy, tqdm"
```
If there are no errors, your installation is complete!

# Usage

## **Basic Usage Example**

This example is for users who want to process a single video with minimal settings:

Process a video with default settings, and save the processed video to `video_output` directory:
`python3 process_video.py -i video.mp4`

Process a video using "window" mode 
`python3 process_video.py -i video.mp4 -m window -ws 20 -bm add -pw 2`