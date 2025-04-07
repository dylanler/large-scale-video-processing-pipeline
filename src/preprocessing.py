import ffmpeg
import logging
import os
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_metadata(video_path):
    """
    Extracts technical metadata from a video file using ffprobe.

    Args:
        video_path (str): The path to the video file.

    Returns:
        dict: A dictionary containing extracted metadata (duration, width, height,
              fps, video_codec, audio_codec, etc.), or None if an error occurs.
    """
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return None
    if os.path.getsize(video_path) == 0:
         logging.warning(f"Video file is empty, cannot extract metadata: {video_path}")
         return None # Cannot process empty files

    try:
        logging.debug(f"Probing video file: {video_path}")
        probe = ffmpeg.probe(video_path)
        
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)

        if not video_stream:
             logging.warning(f"No video stream found in {video_path}")
             # Decide if we want to return partial metadata or None
             # For now, let's return None if there's no video stream
             return None 

        metadata = {
            'filename': os.path.basename(video_path),
            'filepath': video_path,
            'format_name': probe.get('format', {}).get('format_name'),
            'duration_seconds': float(video_stream.get('duration', probe.get('format', {}).get('duration', 0))),
            'width': video_stream.get('width'),
            'height': video_stream.get('height'),
            'avg_frame_rate': eval(video_stream.get('avg_frame_rate', '0/1')), # eval to convert fraction like "30000/1001" to float
            'video_codec': video_stream.get('codec_name'),
            'video_bit_rate': int(video_stream.get('bit_rate', 0)),
            'audio_codec': audio_stream.get('codec_name') if audio_stream else None,
            'audio_channels': audio_stream.get('channels') if audio_stream else None,
            'audio_sample_rate': audio_stream.get('sample_rate') if audio_stream else None,
            'audio_bit_rate': int(audio_stream.get('bit_rate', 0)) if audio_stream else 0,
            'file_size_bytes': int(probe.get('format', {}).get('size', 0))
        }
        
        # Basic quality check examples (can be expanded)
        if metadata['duration_seconds'] < 1.0: # Example: flag very short videos
            logging.debug(f"Video {video_path} is very short ({metadata['duration_seconds']:.2f}s)")
            metadata['quality_flag_short'] = True
        if metadata['width'] is not None and metadata['width'] < 320: # Example: flag low resolution
             logging.debug(f"Video {video_path} has low width ({metadata['width']})")
             metadata['quality_flag_low_res'] = True

        logging.info(f"Successfully extracted metadata for: {os.path.basename(video_path)}")
        return metadata

    except ffmpeg.Error as e:
        stderr_output = e.stderr.decode('utf-8') if e.stderr else "No stderr output"
        logging.error(f"ffmpeg error probing file {video_path}: {stderr_output}")
        # Check for common errors like unsupported format or corrupted file
        if "invalid data found when processing input" in stderr_output:
             logging.error("-> Likely corrupted file or unsupported codec.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error extracting metadata from {video_path}: {e}")
        return None

# --- Main function for testing ---
if __name__ == "__main__":
    # Use videos identified as unique by the deduplication step (if run)
    # For standalone testing, find all videos again
    try:
        from ingestion import find_videos
    except ImportError:
        print("Error: Could not import 'find_videos' from ingestion.py.")
        exit()

    test_input_dir = "data/input_videos"
    output_metadata_dir = "data/metadata_output"

    if not os.path.exists(output_metadata_dir):
        os.makedirs(output_metadata_dir)
        logging.info(f"Created directory: {output_metadata_dir}")

    # Create dummy files if needed for testing structure
    if not os.path.exists(test_input_dir) or not os.listdir(test_input_dir):
         if not os.path.exists(test_input_dir): os.makedirs(test_input_dir)
         dummy_path = os.path.join(test_input_dir, "sample_video_for_meta.mp4")
         if not os.path.exists(dummy_path):
             with open(dummy_path, 'w') as f: pass
             logging.info(f"Created dummy file: {dummy_path}")
         print("\n--- NOTE ---")
         print(f"Using dummy (empty) file in '{test_input_dir}'. Metadata extraction will fail.")
         print("Place actual video files for testing.")
         print("------------\n")


    videos_to_process = find_videos(test_input_dir)
    all_metadata = []

    if not videos_to_process:
        print(f"No videos found in {test_input_dir} to extract metadata from.")
    else:
        print(f"Found {len(videos_to_process)} videos. Extracting metadata...")
        for video_file in videos_to_process:
            print(f"Processing: {os.path.basename(video_file)}")
            metadata = extract_metadata(video_file)
            if metadata:
                all_metadata.append(metadata)
                # Optionally save individual metadata json files
                output_filename = os.path.join(output_metadata_dir,
                                               os.path.splitext(os.path.basename(video_file))[0] + ".json")
                try:
                    with open(output_filename, 'w') as f:
                        json.dump(metadata, f, indent=4)
                    logging.debug(f"Saved metadata to {output_filename}")
                except IOError as e:
                    logging.error(f"Failed to save metadata JSON for {video_file}: {e}")
            else:
                print(f"  -> Failed to extract metadata for {os.path.basename(video_file)}")

    print("\n--- Metadata Extraction Summary ---")
    print(f"Successfully extracted metadata for {len(all_metadata)} out of {len(videos_to_process)} videos.")

    if all_metadata:
        # Optionally save all metadata to a single file (e.g., CSV or JSON lines)
        all_meta_path = os.path.join(output_metadata_dir, "_all_metadata.jsonl")
        try:
            with open(all_meta_path, 'w') as f:
                for meta_record in all_metadata:
                    f.write(json.dumps(meta_record) + '\n')
            print(f"Saved combined metadata to {all_meta_path}")
        except IOError as e:
             logging.error(f"Failed to save combined metadata file: {e}")

        # Example: Print duration of first few videos
        print("\nExample Metadata (Duration):")
        for i, meta in enumerate(all_metadata[:5]):
            print(f"- {meta['filename']}: {meta['duration_seconds']:.2f} seconds") 