import cv2
import numpy as np
import argparse
import os
import sys
from collections import deque
import subprocess
from tqdm import tqdm

def process_video(input_path, output_folder, mode, decay_type, decay_param, blend_mode, bit_depth, window_size, use_ffmpeg, power):
    # Open the input video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Retrieve video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if FPS is not available
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames

    # Set max pixel value based on bit depth
    if bit_depth == 8:
        max_pixel_value = 255
        dtype = np.uint8
    elif bit_depth in [10, 12, 16]:
        max_pixel_value = (2 ** bit_depth) - 1
        dtype = np.uint16
    elif bit_depth == 32:
        max_pixel_value = 1.0  # For float32, values are normalized
        dtype = np.float32
    else:
        print("Error: Unsupported bit depth. Choose 8, 10, 12, 16, or 32.")
        return

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Generate the output filename based on the input parameters
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if mode == 'window':
        mode_str = f"window_{window_size}"
    elif mode == 'decay':
        mode_str = f"{decay_type}_{decay_param}"
    elif mode == 'cumulative':
        mode_str = "cumulative"
    else:
        mode_str = "unknown_mode"
    
    output_filename = f"{base_name}_{mode_str}_{blend_mode}_power_{power}_{bit_depth}bit.mp4"
    output_path = os.path.join(output_folder, output_filename)

    # If file already exists, append a counter (e.g., _2, _3)
    count = 2
    while os.path.exists(output_path):
        output_path = os.path.join(output_folder, f"{os.path.splitext(output_filename)[0]}_{count}.mp4")
        count += 1

    print(f"Saving output to: {output_path}")

    # Create output directory for frames if bit depth is greater than 8
    if bit_depth > 8:
        frames_dir = os.path.splitext(output_path)[0] + "_frames"
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Frames will be saved to directory '{frames_dir}'")

    # Print starting message with options
    print(f"Starting processing '{os.path.basename(input_path)}' ({width}x{height}, {frame_count} frames, {bit_depth}-bit)")
    if bit_depth == 8:
        print(f"Output will be saved to '{output_path}'")
    print("Processing options:")
    print(f"  Mode: {mode}")
    if mode == 'window':
        print(f"  Window size: {window_size}")
    elif mode == 'decay':
        print(f"  Decay type: {decay_type}")
        print(f"  Decay parameter: {decay_param}")
    print(f"  Blend mode: {blend_mode}")
    print(f"  Bit depth: {bit_depth}")
    if use_ffmpeg:
        print(f"  FFmpeg HDR ProRes export: Enabled")
    else:
        print(f"  FFmpeg HDR ProRes export: Disabled")
    print("Processing...")

    # Initialize video writer for 8-bit output
    if bit_depth == 8:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize processing variables
    frame_number = 0  # Current frame number
    cumulative_frame = None
    frame_buffer = deque(maxlen=window_size) if mode == 'window' else None

    for frame_number in tqdm(range(frame_count), desc="Processing Frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no frames are left to read

        frame = frame.astype(dtype)
        if bit_depth == 32:
            frame = frame / 255.0  # Normalize to 0.0 - 1.0

        frame_float = frame.astype(np.float32)

        # Scale pixel values for HDR modes
        if bit_depth in [10, 12, 16]:
            frame_float = frame_float * (max_pixel_value / 255.0)

        if mode == 'window':
            # Window size mode
            frame_buffer.append(frame_float.copy())
            if len(frame_buffer) == window_size:
                # Combine frames in the buffer
                cumulative_frame = frame_buffer[0].copy()
                for idx in range(1, window_size):
                    cumulative_frame = blend_frames(cumulative_frame, frame_float, blend_mode, max_pixel_value, power)
        elif mode == 'decay':
            # Decay mode
            if cumulative_frame is None:
                # Initialize cumulative_frame with the first frame
                cumulative_frame = frame_float
            else:
                # Apply decay
                if decay_type == 'exponential':
                    cumulative_frame *= decay_param
                elif decay_type == 'linear':
                    if bit_depth == 32:
                        cumulative_frame -= decay_param / 255.0
                    else:
                        cumulative_frame -= decay_param * (max_pixel_value / 255.0)
                else:
                    print("Error: Invalid decay type.")
                    cap.release()
                    if bit_depth == 8:
                        out.release()
                    return
                # Ensure cumulative_frame stays within valid range
                cumulative_frame = np.clip(cumulative_frame, 0, max_pixel_value)

                # Apply blend mode
                cumulative_frame = blend_frames(cumulative_frame, frame_float, blend_mode, max_pixel_value, power)
        elif mode == 'cumulative':
            # Cumulative mode
            if cumulative_frame is None:
                # Initialize cumulative_frame with the first frame
                cumulative_frame = frame_float
            else:
                # Apply blend mode directly
                cumulative_frame = blend_frames(cumulative_frame, frame_float, blend_mode, max_pixel_value, power)
        else:
            print("Error: Invalid mode.")
            cap.release()
            if bit_depth == 8:
                out.release()
            return

        # Convert cumulative frame to appropriate data type
        if cumulative_frame is not None:
            cumulative_frame = np.clip(cumulative_frame, 0, max_pixel_value)
            if bit_depth == 8:
                output_frame = cumulative_frame.astype(np.uint8)
            elif bit_depth in [10, 12, 16]:
                output_frame = cumulative_frame.astype(np.uint16)
            elif bit_depth == 32:
                output_frame = cumulative_frame.astype(np.float32)

            # Write the processed frame to the output
            if bit_depth == 8:
                out.write(output_frame)
            else:
                # Save frame as image in output directory
                frame_filename = os.path.join(output_path, f"frame_{frame_number:06d}.png")
                if bit_depth == 32:
                    # For float32, save as 32-bit TIFF
                    frame_filename = os.path.join(output_path, f"frame_{frame_number:06d}.tiff")
                    cv2.imwrite(frame_filename, output_frame, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
                else:
                    # For uint16, save as 16-bit PNG
                    cv2.imwrite(frame_filename, output_frame)

        frame_number += 1

    # Release video resources
    cap.release()
    if bit_depth == 8:
        out.release()

    # Print completion message
    print(f"Processing completed.")
    if bit_depth == 8:
        print(f"Wrote '{output_path}' ({width}x{height}, {frame_number} frames, {bit_depth}-bit) with options:")
    else:
        print(f"Saved frames to '{output_path}' ({width}x{height}, {frame_number} frames, {bit_depth}-bit) with options:")
    print(f"  Mode: {mode}")
    if mode == 'window':
        print(f"  Window size: {window_size}")
    elif mode == 'decay':
        print(f"  Decay type: {decay_type}")
        print(f"  Decay parameter: {decay_param}")
    print(f"  Blend mode: {blend_mode}")
    print(f"  Bit depth: {bit_depth}")

    # Optional FFmpeg HDR ProRes export
    if use_ffmpeg and bit_depth >= 10:
        print("Starting FFmpeg HDR ProRes export...")
        ffmpeg_export(output_path, fps)
    elif use_ffmpeg:
        print("FFmpeg HDR ProRes export is only supported for bit depths >= 10.")

def ffmpeg_export(frames_directory, fps):
    # Construct FFmpeg command
    # Assuming frames are saved as frame_000000.png, frame_000001.png, etc.
    input_pattern = os.path.join(frames_directory, 'frame_%06d.png')
    output_file = frames_directory.rstrip('_frames') + '_prores.mov'

    ffmpeg_command = [
        'ffmpeg',
        '-r', str(fps),
        '-f', 'image2',
        '-i', input_pattern,
        '-c:v', 'prores_ks',
        '-profile:v', '3',
        '-pix_fmt', 'yuv422p10le',
        '-color_primaries', '9',
        '-color_trc', '16',
        '-colorspace', '9',
        output_file
    ]

    print(f"Executing FFmpeg command: {' '.join(ffmpeg_command)}")
    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"FFmpeg HDR ProRes export completed. Output file: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg HDR ProRes export failed with error: {e}")

def blend_frames(cumulative_frame, frame_float, blend_mode, max_pixel_value, power=1):
    """
    Blend frames using the specified blend mode, with an optional power dropoff factor.
    The power parameter controls how much the blend effect diminishes as the pixel value moves
    away from the maximum possible value.
    """
    # Normalize frame values to [0, 1] for easier calculations
    normalized_frame = frame_float / max_pixel_value
    normalized_cumulative = cumulative_frame / max_pixel_value

    intensity_factor = np.power(normalized_frame, power)

    if blend_mode == 'lighten':
        # Lighten blend: use the maximum, but modulate based on power
        result = np.maximum(cumulative_frame, frame_float)
        return cumulative_frame + (result - cumulative_frame) * intensity_factor

    elif blend_mode == 'screen':
        # Standard screen blend formula with diminishing effect controlled by the power function
        screen_result = max_pixel_value - ((max_pixel_value - cumulative_frame) * (max_pixel_value - frame_float) / max_pixel_value)
        return cumulative_frame + (screen_result - cumulative_frame) * intensity_factor

    elif blend_mode == 'add':
        # Add blend mode with power-based diminishing effect
        return np.clip(cumulative_frame + frame_float * intensity_factor, 0, max_pixel_value)

    elif blend_mode == 'multiply':
        # Multiply blend modulated by power
        return np.clip((cumulative_frame * frame_float) ** power / max_pixel_value, 0, max_pixel_value)

    elif blend_mode == 'overlay':
        # Overlay blend mode with power modulating the effect
        mask = cumulative_frame <= (max_pixel_value / 2)
        result = np.empty_like(cumulative_frame)
        result[mask] = (2 * cumulative_frame[mask] * frame_float[mask]) / max_pixel_value
        result[~mask] = max_pixel_value - 2 * (max_pixel_value - cumulative_frame[~mask]) * (max_pixel_value - frame_float[~mask]) / max_pixel_value
        return cumulative_frame + (result - cumulative_frame) * intensity_factor

    elif blend_mode == 'soft_light':
        # Soft Light with power modulation
        soft_light_result = (1 - (1 - cumulative_frame / max_pixel_value) * (1 - frame_float / max_pixel_value)) * max_pixel_value
        return cumulative_frame + (soft_light_result - cumulative_frame) * intensity_factor

    elif blend_mode == 'hard_light':
        # Hard Light with power modulation
        mask = frame_float <= (max_pixel_value / 2)
        result = np.empty_like(cumulative_frame)
        result[mask] = (2 * cumulative_frame[mask] * frame_float[mask]) / max_pixel_value
        result[~mask] = max_pixel_value - 2 * (max_pixel_value - cumulative_frame[~mask]) * (max_pixel_value - frame_float[~mask]) / max_pixel_value
        return cumulative_frame + (result - cumulative_frame) * intensity_factor

    elif blend_mode == 'difference':
        # Difference blend with power modulation
        difference_result = np.abs(cumulative_frame - frame_float)
        return cumulative_frame + (difference_result - cumulative_frame) * intensity_factor

    elif blend_mode == 'exclusion':
        # Exclusion blend with power modulation
        exclusion_result = cumulative_frame + frame_float - (2 * cumulative_frame * frame_float) / max_pixel_value
        return cumulative_frame + (exclusion_result - cumulative_frame) * intensity_factor

    elif blend_mode == 'color_dodge':
        # Color Dodge with power modulation
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.minimum(max_pixel_value, (cumulative_frame * max_pixel_value) / (max_pixel_value - frame_float + 1e-5))
        result[np.isnan(result)] = max_pixel_value
        return cumulative_frame + (result - cumulative_frame) * intensity_factor

    elif blend_mode == 'color_burn':
        # Color Burn with power modulation
        with np.errstate(divide='ignore', invalid='ignore'):
            result = max_pixel_value - np.minimum(max_pixel_value, (max_pixel_value - cumulative_frame) * max_pixel_value / (frame_float + 1e-5))
        result[np.isnan(result)] = 0
        return cumulative_frame + (result - cumulative_frame) * intensity_factor

    elif blend_mode == 'linear_dodge':
        # Linear Dodge (Add) with power modulation
        return np.clip(cumulative_frame + frame_float * intensity_factor, 0, max_pixel_value)

    elif blend_mode == 'linear_burn':
        # Linear Burn with power modulation
        return np.clip(cumulative_frame + (frame_float - max_pixel_value) * intensity_factor, 0, max_pixel_value)

    else:
        raise ValueError("Invalid blend mode")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video processing with blend modes, bit depth, decay, and more.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input video file.")
    parser.add_argument("-o", "--output_folder", default="video_output", help="Directory to save the output videos.")
    parser.add_argument("-m", "--mode", choices=['cumulative', 'decay', 'window'], default='decay', help="Processing mode ('cumulative', 'decay', 'window').")
    parser.add_argument("-dt", "--decay_type", choices=['exponential', 'linear'], default='exponential', help="Decay type ('exponential' or 'linear') for decay mode.")
    parser.add_argument("-dp", "--decay_param", type=float, default=0.95, help="Decay parameter for decay mode.")
    parser.add_argument("-ws", "--window_size", type=int, help="Window size for window mode.")
    parser.add_argument("-bm", "--blend_mode", choices=[
        'lighten', 'screen', 'add', 'multiply', 'overlay', 'soft_light', 'hard_light', 'difference', 'exclusion', 'color_dodge', 'color_burn', 'linear_dodge', 'linear_burn'
    ], default='lighten', help="Blend mode.")
    parser.add_argument("-bd", "--bit_depth", type=int, choices=[8, 10, 12, 16, 32], default=8, help="Bit depth (8, 10, 12, 16, or 32).")
    parser.add_argument("-ff", "--use_ffmpeg", action='store_true', help="Use FFmpeg to export HDR ProRes video (experimental, for bit depths >=10).")
    parser.add_argument("-pw", "--power", type=float, default=1.0, help="Power dropoff for blend effect. Controls how the blend diminishes as values move away from the maximum (default=1 for linear dropoff).")

    args = parser.parse_args()

    # Initialize parameters
    decay_type = None
    decay_param = None
    window_size = None

    # Validate parameters based on mode
    if args.mode == 'window':
        if args.window_size is None or args.window_size < 1:
            print("Error: Window size must be a positive integer for window mode.")
            sys.exit(1)
        window_size = args.window_size
    elif args.mode == 'decay':
        if args.decay_type == 'exponential':
            if not (0 < args.decay_param < 1):
                print("Error: For exponential decay, decay factor must be between 0 and 1 (non-inclusive).")
                sys.exit(1)
        elif args.decay_type == 'linear':
            if args.decay_param < 0:
                print("Error: For linear decay, decay amount must be non-negative.")
                sys.exit(1)
        else:
            print("Error: Invalid decay type.")
            sys.exit(1)
        decay_type = args.decay_type
        decay_param = args.decay_param
    elif args.mode == 'cumulative':
        # No additional parameters needed
        pass
    else:
        print("Error: Invalid mode selected.")
        sys.exit(1)

    process_video(args.input, args.output_folder, args.mode, args.decay_type, args.decay_param, args.blend_mode, args.bit_depth, args.window_size, args.use_ffmpeg, args.power)
