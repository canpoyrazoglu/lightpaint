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

## Output resolution

Controls the output video resolution. Used with `--output_resolution` (or `-r`) parameter. Can be used like `-r 1920x1080` to specify width and height respectively.

##

Controls the output frame rate. Used with `--output_fps` (or `-fps`) parameter. It does _not_ smart-blend the images to keep the video time in sync with input, it just specifies the output frame rate. So, for example, if input video is 60FPS and you specify 30FPS, output will be slowed down 2x naturally.

## Bit depth

_WARNING: This parameter is experimental and may not work in many scenarios!_

Controls the bit depth of processing and output. Used with `--bit_depth` or `-bd` parameter. Default is `8` (SDR) and allowed values are `16` and `32`, which they all act as HDR mode. In HDR mode, single frames are saved as output instead of a video file, _unless `--use_ffmpeg` flag is used_. Please note that ffmpeg HDR output works in very few scenarios and fails in most scenarios. 

## Use ffmpeg

Controls whether or not to use ffmpeg with 16-bit bit depths for HDR ProRes video output. Experimental and works with very specific scenarios only. Used with `--use-ffmpeg` (or `-ff`),

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
pip3 install opencv-python numpy tqdm
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

## Usage for single output

To process a video, `process_video.py` script is used. Here's an example is for users who want to process a single video with minimal settings:

Process a video with default settings, and save the processed video to `video_output` directory:
`python3 process_video.py -i video.mp4`

Process a video using "window" mode, with window size of 20 frames, blending mode `add` and a power dropoff parameter of `2`:
`python3 process_video.py -i video.mp4 -m window -ws 20 -bm add -pw 2`

Process a video with `screen` blend mode and decay value of `0.9`:
`python3 process_video.py -i video.mp4 -bm screen -dp 0.90`

## Batch processing

Sometimes it's hard to imagine how things will turn out before creating the final video, and you might want to experiment with a range of settings. Batch processing does exactly that. Imagine you have an input video and you want to try different blend modes, decay parameters, and power dropoff values to see what results in the style you'd want the most. With batch processing script, you can create combinations of outputs based on the parameters that accept multiple values. 

For batch processing, `lightpaint_batch.py` script is used. Here is an example (script converted to multiline for readibility, you can of course type all in a single line):

```
python3 lightpaint_batch.py \        
    -i paint.mp4 \
    -bm lighten,add,screen \
    -pw 1,1.2,1.5,2 \
    -dp 0.6,0.7,0.8,0.9,0.95,0.98
```

This script takes an input video `paint.mp4`, and creates different versions of output with different blend modes, power dropoff, and decay parameters. It will create every combination of blend modes (`lighten`, `add`, `screen`), power dropoff values (`1`, `1.2`, `1.5`, `2`), and decay parameters (`0.6`, `0.7`, `0.8`, `0,9`, `0.95`, `0.98`). So it will create 3 (blend modes) × 4 (power dropoff values) × 6 (decay parameters) = 72 videos. Batch processing script spawns a new instance of the original script based on your device's CPU cores so, if you have 8 cores, it can process 8 videos at once for a faster batch output.

Note: Batch processing parameters are currently limited to decay, power, and blending mode variations. Specifying mode (therefore window size), decay type, and bit depth are currently not supported.

# Examples

I've ran the batch processing script on the reference video, `input.mp4`, and the example above, which created 72 variations of the video. You can check them out at `video_output` directory.

# Fun fact

I'm not a Python developer, and I've used GPT-4o and o1-preview to create this script. I first started a conversation with 4o, which, in the meantime, o1-preview was announced where I had the very basic script ready. If you want to dive into the process of how I "created" this from scratch, you can read the whole conversation with all the pitfalls, strong sides of code generation, and problems I faced on the following link: https://chatgpt.com/share/66e5ba82-57a8-8005-9731-db928a795a85 (it doesn't contain the initial conversation of creating the first script, but it contains the iterative process of adding/changing features since then). I made little modifications but the general structure is the same.

# Limitations

The script is available as-is and is definitely not perfect. It started as a tool for myself where I needed to process some experimental videos of lightpainting and I wanted to grow it into something modular. Because of my limited time, little actual Python knowledge, and priorities, it has its limitations:

 - The processing runs on the CPU, single threaded. It _is_ slow. Acceptable for most videos under 1 minutes, but definitely not fast. I'm not sure, because of nature of the script depending on previous frames' input, whether it can ever be made multithreaded or be moved to the GPU, nor I have any plans to do so. If you are a developer who wants to go that rabbit hole, feel free to do so :)

 - ffmpeg and HDR supportis _very_ limited. I could only create a few ProRes HDR videos and that's it. I didn't have any luck with H.264 or H.265 regarding HDR. I'd love to find a solution but I'm not an ffmpeg guru, so if you have extensive knowledge of ffmpeg, codes, color spaces, gamma/log functions, bit depth, and any other video encoding parameters related to HDR especially in the context of ffmpeg, please reach out with a working example <3.

 - Many combinations of blend modes and different parameters have not been tried and may produce inferiour results. Though with some experimentation, I believe there might be use cases for "non-brightening" (basically anything other than `lighten`, `screen`, and `add`) blend modes, especially with `cumulative` or `window` modes.

 - `window` mode is slow. I repeat: `window` mode _is_ slow. Espcially with large window sizes.

 - I tested this only on a MacBook Pro (M1 Max, 64GB RAM, macOS 14+15) and don't know how it would work with other systems.