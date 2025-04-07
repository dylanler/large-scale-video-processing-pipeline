import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_videos(input_dir):
    """
    Finds video files in the specified input directory.
    For local testing, this replaces the web crawling component.

    Args:
        input_dir (str): The directory to search for video files.

    Returns:
        list: A list of paths to video files found.
              Returns an empty list if the directory doesn't exist or contains no videos.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv') # Add more if needed
    video_files = []

    if not os.path.isdir(input_dir):
        logging.warning(f"Input directory not found: {input_dir}")
        return video_files

    logging.info(f"Scanning for videos in: {input_dir}")
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_path = os.path.join(root, file)
                video_files.append(video_path)
                logging.debug(f"Found video: {video_path}")

    logging.info(f"Found {len(video_files)} video files.")
    return video_files

if __name__ == "__main__":
    # Example usage for local testing
    # Assumes running from the root directory, adjust path if running from src
    test_input_dir = "data/input_videos" 

    # Create dummy input dir and a file for testing if they don't exist
    if not os.path.exists(test_input_dir):
        os.makedirs(test_input_dir)
        # Create a dummy empty file to simulate a video
        dummy_video_path = os.path.join(test_input_dir, "sample_video.mp4")
        if not os.path.exists(dummy_video_path):
             with open(dummy_video_path, 'w') as f:
                pass # Create an empty file
             logging.info(f"Created dummy directory and file for testing: {dummy_video_path}")


    videos_to_process = find_videos(test_input_dir)
    if videos_to_process:
        print("Videos found:")
        for video in videos_to_process:
            print(f"- {video}")
    else:
        print(f"No videos found in {test_input_dir}. Please add some sample videos.") 