import os
import logging
import cv2 # Using opencv-python-headless
import imagehash
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_sample_frames(video_path, num_frames=5):
    """Extracts a small number of evenly spaced frames from a video."""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Cannot open video file: {video_path}")
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Handle videos with very few frames or potential read errors
        if total_frames <= 0:
             logging.warning(f"Video {video_path} reported 0 or negative frames. Trying to read first frame.")
             # Try reading the first frame directly
             ret, frame = cap.read()
             if ret:
                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                 frames.append(Image.fromarray(frame_rgb))
                 logging.info(f"Successfully read the first frame of {video_path} despite frame count issue.")
             else:
                 logging.error(f"Could not read even the first frame from {video_path}")
             cap.release()
             return frames

        # Adjust num_frames if video is shorter
        if total_frames < num_frames:
            logging.debug(f"Video {video_path} has fewer ({total_frames}) than requested {num_frames} frames. Using all available.")
            num_frames = total_frames # Sample all frames if fewer than requested

        # Ensure num_frames is at least 1 if total_frames > 0
        if num_frames == 0 and total_frames > 0:
            num_frames = 1

        # Generate frame indices
        if num_frames > 0:
             # Use linspace to get evenly spaced indices including the first and potentially last frame
             frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
             # Ensure uniqueness, especially for short videos
             frame_indices = np.unique(frame_indices) 

             for i in frame_indices:
                 cap.set(cv2.CAP_PROP_POS_FRAMES, float(i)) # Use float for robustness with some backends
                 ret, frame = cap.read()
                 if ret:
                     # Convert frame BGR (OpenCV default) to RGB (PIL/imagehash default)
                     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                     frames.append(Image.fromarray(frame_rgb))
                 else:
                     # Try reading the next available frame if the exact index failed
                     logging.warning(f"Could not read frame index {i} from {video_path}. Trying subsequent frame.")
                     ret, frame = cap.read() # Read the very next frame
                     if ret:
                         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                         frames.append(Image.fromarray(frame_rgb))
                     else:
                         logging.error(f"Still could not read frame near index {i} from {video_path}")

        cap.release()
    except Exception as e:
        logging.error(f"Error extracting frames from {video_path}: {e}", exc_info=True)
        if 'cap' in locals() and cap.isOpened():
            cap.release() # Ensure cleanup on error
    return frames

def calculate_video_phashes(video_path, num_frames=5, hash_size=8):
    """
    Calculates perceptual hashes (pHash) for sample frames of a video.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to sample.
        hash_size (int): The size of the pHash (e.g., 8 -> 8x8 hash).

    Returns:
        list: A list of imagehash.ImageHash objects, or None if hashing failed.
    """
    hashes = []
    frames = extract_sample_frames(video_path, num_frames)
    if not frames:
        logging.warning(f"No frames extracted from {video_path}, cannot calculate phash.")
        return None

    try:
        for frame in frames:
            hashes.append(imagehash.phash(frame, hash_size=hash_size))
        logging.debug(f"Calculated {len(hashes)} pHashes for {video_path}")
    except Exception as e:
        logging.error(f"Error calculating phash for frames from {video_path}: {e}")
        return None
    return hashes


def find_duplicate(video_path, existing_hashes_db, num_frames=5, hash_size=8, threshold=5):
    """
    Checks if a video is likely a duplicate based on perceptual hashes
    of its frames compared to a database of existing hashes.

    Args:
        video_path (str): Path to the video file to check.
        existing_hashes_db (dict): A dictionary where keys are video IDs/paths
                                   and values are lists of their frame phashes.
        num_frames (int): Number of frames to sample for hashing.
        hash_size (int): Size of the perceptual hash.
        threshold (int): Maximum average Hamming distance to consider videos similar.

    Returns:
        tuple: (str, float) -> (path_of_duplicate, average_distance) if a duplicate is found,
               otherwise None.
    """
    current_phashes = calculate_video_phashes(video_path, num_frames, hash_size)
    if not current_phashes:
        logging.warning(f"Could not calculate phashes for {video_path}. Skipping duplicate check.")
        # Returning None indicates we couldn't determine duplication status
        return None

    min_avg_distance = float('inf')
    closest_match_id = None

    for existing_id, existing_phashes in existing_hashes_db.items():
        # Basic check: skip if hash list is empty or length differs significantly
        # (A more robust check might handle videos of different lengths/framerate better)
        if not existing_phashes or len(existing_phashes) != len(current_phashes):
            continue

        total_distance = 0
        valid_comparisons = 0
        for h1, h2 in zip(current_phashes, existing_phashes):
            # Ensure both hashes are valid ImageHash objects
            if isinstance(h1, imagehash.ImageHash) and isinstance(h2, imagehash.ImageHash):
                try:
                    total_distance += (h1 - h2) # Hamming distance calculation
                    valid_comparisons += 1
                except Exception as e:
                     logging.error(f"Error comparing hashes {h1} and {h2}: {e}")
                     # Decide how to handle comparison errors, e.g., skip this pair
                     continue # Skip this pair if comparison fails

        # Only calculate average if we successfully compared all pairs
        if valid_comparisons == len(current_phashes) and valid_comparisons > 0 :
            average_distance = total_distance / valid_comparisons
            logging.debug(f"Comparing {os.path.basename(video_path)} with {os.path.basename(existing_id)}: Avg dist={average_distance:.2f}")

            # Check if this is a potential match below the threshold
            if average_distance <= threshold:
                 # If it's closer than previous matches, update
                 if average_distance < min_avg_distance:
                    min_avg_distance = average_distance
                    closest_match_id = existing_id
                    logging.info(f"Found potential duplicate for {video_path}: {existing_id} with avg distance {average_distance:.2f}")

    if closest_match_id is not None:
        return (closest_match_id, min_avg_distance)

    # No duplicate found below the threshold
    return None


# --- Main function for testing ---
if __name__ == "__main__":
    # Assumes ingestion.py is in the same directory or Python path
    try:
        from ingestion import find_videos
    except ImportError:
        print("Error: Could not import 'find_videos' from ingestion.py.")
        print("Make sure ingestion.py is in the same directory or your PYTHONPATH.")
        exit()

    test_input_dir = "data/input_videos"
    processed_video_hashes = {} # Store hashes of non-duplicate videos found so far
    unique_videos = []
    duplicate_videos = {} # Store mapping from duplicate path to original path

    # Ensure the test directory exists and has dummy files if needed
    if not os.path.exists(test_input_dir):
        os.makedirs(test_input_dir)
        logging.info(f"Created directory: {test_input_dir}")
        # Create dummy files for basic structure testing
        for i in range(3):
             dummy_path = os.path.join(test_input_dir, f"sample_video_{i+1}.mp4")
             if not os.path.exists(dummy_path):
                 with open(dummy_path, 'w') as f: pass
                 logging.info(f"Created dummy file: {dummy_path}")
        print("\n--- NOTE ---")
        print("Using dummy (empty) video files for testing structure.")
        print(f"Place actual video files (some identical, some different) in '{test_input_dir}' for meaningful deduplication testing.")
        print("For example, copy one video to 'vid1.mp4' and 'vid2_dup.mp4', and use a different one for 'vid3_diff.mp4'.")
        print("------------\n")


    # 1. Find all videos using the ingestion module
    all_videos = find_videos(test_input_dir)

    if not all_videos:
        print(f"No video files found in {test_input_dir}. Exiting.")
        exit()

    print(f"Found {len(all_videos)} videos. Checking for duplicates...")

    # 2. Iterate and check for duplicates
    for video_file in all_videos:
        if not os.path.exists(video_file) or os.path.getsize(video_file) == 0:
             logging.warning(f"Skipping empty or non-existent file: {video_file}")
             continue

        print(f"Processing: {os.path.basename(video_file)}")
        duplicate_info = find_duplicate(video_file, processed_video_hashes, threshold=5) # Lower threshold means stricter match

        if duplicate_info:
            original_id, distance = duplicate_info
            print(f"  -> Result: DUPLICATE of {os.path.basename(original_id)} (Distance: {distance:.2f})")
            duplicate_videos[video_file] = original_id
        else:
            # If not a duplicate, calculate hashes (if not already done by find_duplicate)
            # and add to the database for future comparisons.
            current_phashes = calculate_video_phashes(video_file)
            if current_phashes:
                 print(f"  -> Result: UNIQUE (added to known videos)")
                 processed_video_hashes[video_file] = current_phashes
                 unique_videos.append(video_file)
            else:
                 print(f"  -> Result: ERROR processing (could not get hash)")
                 # Handle videos that couldn't be processed - maybe add to a separate list


    print("\n--- Deduplication Summary ---")
    print(f"Total videos found: {len(all_videos)}")
    print(f"Unique videos to process: {len(unique_videos)}")
    print(f"Duplicate videos found: {len(duplicate_videos)}")

    if unique_videos:
        print("\nUnique Videos:")
        for vid in unique_videos:
            print(f"- {os.path.basename(vid)}")

    if duplicate_videos:
        print("\nDuplicate Mappings (Duplicate -> Original):")
        for dup, orig in duplicate_videos.items():
            print(f"- {os.path.basename(dup)} -> {os.path.basename(orig)}") 