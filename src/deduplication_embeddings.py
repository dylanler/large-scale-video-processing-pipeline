import os\nimport logging\nimport cv2\nfrom PIL import Image\nimport numpy as np\nimport torch\nfrom sentence_transformers import SentenceTransformer\nfrom sklearn.metrics.pairwise import cosine_similarity\nimport time\n\n# Import frame extraction from the other deduplication module\n# In a larger project, this might be moved to a shared utils module\ntry:\n    from deduplication import extract_sample_frames\nexcept ImportError:\n     logging.error(\"Could not import extract_sample_frames. Make sure deduplication.py is accessible.\")\n     # Define a dummy function if import fails, to allow script loading\n     def extract_sample_frames(video_path, num_frames=5):\n         logging.warning(\"Using dummy extract_sample_frames due to import error.\")\n         return []\n\nlogging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(name)s - %(message)s\')\nlogger = logging.getLogger(__name__)\n\n# --- Global Model Loading ---\n# Load the model only once when the module is imported for efficiency.\n# Using a smaller CLIP model for faster local testing without GPU.\n# For production, use a larger model and run on GPU.\nMODEL_NAME = \'clip-vit-base-patch32\'\nMODEL = None\nDEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n\ndef load_embedding_model():\n    global MODEL\n    if MODEL is None:\n        logger.info(f\"Loading embedding model ({MODEL_NAME}) onto {DEVICE}...\")\n        start_load = time.time()\n        try:\n            MODEL = SentenceTransformer(MODEL_NAME, device=DEVICE)\n            logger.info(f\"Model loaded in {time.time() - start_load:.2f} seconds.\")\n        except Exception as e:\n            logger.error(f\"Failed to load SentenceTransformer model \'{MODEL_NAME}\': {e}\", exc_info=True)\n            MODEL = None # Ensure model remains None if loading failed\n    return MODEL\n\n# --- Embedding Calculation ---\ndef calculate_video_embeddings(video_path, model, num_frames=5):\n    \"\"\"\n    Calculates embeddings for sample frames of a video using a SentenceTransformer model.\n\n    Args:\n        video_path (str): Path to the video file.\n        model (SentenceTransformer): The loaded embedding model.\n        num_frames (int): Number of frames to sample.\n\n    Returns:\n        np.ndarray or None: A numpy array where each row is an embedding for a frame,\n                            or None if embedding calculation failed.\n    \"\"\"\n    if model is None:\n        logger.error(\"Embedding model is not loaded. Cannot calculate embeddings.\")\n        return None\n\n    embeddings = []\n    frames = extract_sample_frames(video_path, num_frames)\n    if not frames:\n        logger.warning(f\"No frames extracted from {video_path}, cannot calculate embeddings.\")\n        return None\n\n    try:\n        # Batch encode PIL images directly (supported by SentenceTransformer)\n        logger.debug(f\"Encoding {len(frames)} frames from {os.path.basename(video_path)}...\")\n        frame_embeddings = model.encode(frames, convert_to_numpy=True, show_progress_bar=False, device=DEVICE)\n        logger.debug(f\"Calculated {len(frame_embeddings)} embeddings for {os.path.basename(video_path)}\")\n        # Return as a single NumPy array (num_frames x embedding_dim)\n        return frame_embeddings\n    except Exception as e:\n        logger.error(f\"Error calculating embeddings for frames from {video_path}: {e}\", exc_info=True)\n        return None\n\n# --- Duplicate Detection --- \ndef find_duplicate_by_embedding(video_path, model, existing_embeddings_db, num_frames=5, threshold=0.95):\n    \"\"\"\n    Checks if a video is likely a duplicate based on embedding similarity.\n\n    Args:\n        video_path (str): Path to the video file to check.\n        model (SentenceTransformer): The loaded embedding model.\n        existing_embeddings_db (dict): {video_id: np.ndarray_of_frame_embeddings}\n        num_frames (int): Number of frames to sample.\n        threshold (float): Minimum average cosine similarity to be considered a duplicate.\n\n    Returns:\n        tuple: (str, float) -> (path_of_duplicate, average_similarity) if duplicate found,\n               otherwise None.\n    \"\"\"\n    current_embeddings = calculate_video_embeddings(video_path, model, num_frames)\n    if current_embeddings is None or current_embeddings.shape[0] == 0:\n        logger.warning(f\"Could not calculate embeddings for {video_path}. Skipping duplicate check.\")\n        return None\n\n    max_avg_similarity = -1.0\n    closest_match_id = None\n\n    for existing_id, existing_embeddings in existing_embeddings_db.items():\n        # Check if shapes match (same number of frames sampled)\n        if existing_embeddings is None or current_embeddings.shape != existing_embeddings.shape:\n             logger.debug(f\"Embedding shape mismatch comparing {os.path.basename(video_path)} ({current_embeddings.shape}) and {os.path.basename(existing_id)} ({existing_embeddings.shape if existing_embeddings is not None else \'None\'}). Skipping.\")\n             continue\n\n        try:\n            # Calculate cosine similarity between corresponding frame embeddings\n            similarities = cosine_similarity(current_embeddings, existing_embeddings)\n            # We want the diagonal elements, as we compare frame 1 to frame 1, frame 2 to frame 2, etc.\n            pair_similarities = np.diag(similarities)\n            \n            if len(pair_similarities) > 0:\n                average_similarity = np.mean(pair_similarities)\n                logger.debug(f\"Comparing {os.path.basename(video_path)} with {os.path.basename(existing_id)}: Avg similarity={average_similarity:.4f}\")\n\n                if average_similarity >= threshold:\n                    if average_similarity > max_avg_similarity:\n                        max_avg_similarity = average_similarity\n                        closest_match_id = existing_id\n                        logger.info(f\"Found potential duplicate for {video_path}: {existing_id} with avg similarity {average_similarity:.4f}\")\n        except Exception as e:\
            logger.error(f\"Error calculating similarity between {video_path} and {existing_id}: {e}\", exc_info=True)\n            continue\n            \n    if closest_match_id is not None:\n        return (closest_match_id, max_avg_similarity)\n\n    return None\n\n# --- Main function for testing ---\nif __name__ == \"__main__\":\n    # This test requires actual video files for meaningful results.\n    # Dummy files won't work as frame extraction will fail.\n    try:\n        from ingestion import find_videos\n    except ImportError:\n        print(\"Error: Could not import \'find_videos\' from ingestion.py.\")\n        exit()\n\n    # Load the model first\n    model = load_embedding_model()\n    if model is None:\n         print(\"Failed to load the embedding model. Cannot run deduplication.\")\n         exit()\n\n    test_input_dir = \"data/input_videos\"\n    processed_video_embeddings = {} # Store embeddings {path: np.array}\n    unique_videos = []\n    duplicate_videos = {} # Store mapping {duplicate_path: original_path}\n\n    # --- NOTE --- \n    print(\"\\n--- NOTE ---\")\n    print(\"Running EMBEDDING-based deduplication test.\")\n    print(f\"This requires actual video files in \'{test_input_dir}\'.\")\n    print(\"Place some identical videos and some different ones.\")\n    print(\"Model used: \", MODEL_NAME)\n    print(\"Device: \", DEVICE)\n    print(\"------------\\n\")\n    # Optionally create dummy *directory* if needed, but files must be real\n    if not os.path.exists(test_input_dir):\n        os.makedirs(test_input_dir)\n        print(f\"Created directory {test_input_dir}. Please add videos.\")\n\n    # 1. Find videos\n    all_videos = find_videos(test_input_dir)\n    valid_videos = [f for f in all_videos if os.path.exists(f) and os.path.getsize(f) > 0]\n\n    if not valid_videos:\n        print(f\"No valid video files found in {test_input_dir}. Exiting.\")\n        exit()\n\n    print(f\"Found {len(valid_videos)} valid videos. Calculating embeddings and checking duplicates...\")\n    similarity_threshold = 0.95 # High threshold for likely duplicates\n\n    # 2. Iterate and check duplicates\n    for video_file in valid_videos:\n        print(f\"Processing: {os.path.basename(video_file)}\")\n        # Calculate embeddings for the current video\n        current_embeddings = calculate_video_embeddings(video_file, model)\n        \n        if current_embeddings is None:\n            print(f\"  -> Result: ERROR processing (could not get embeddings)\")\n            continue # Skip this video if embeddings failed\n\n        # Find duplicate by comparing with the existing DB\n        duplicate_info = find_duplicate_by_embedding(video_file, model, processed_video_embeddings, threshold=similarity_threshold)\n\n        if duplicate_info:\n            original_id, similarity = duplicate_info\n            print(f\"  -> Result: DUPLICATE of {os.path.basename(original_id)} (Similarity: {similarity:.4f})\")\n            duplicate_videos[video_file] = original_id\n        else:\n            # Not a duplicate (or embeddings failed for comparison video), add to DB\n            print(f\"  -> Result: UNIQUE (added to known videos)\")\n            processed_video_embeddings[video_file] = current_embeddings\n            unique_videos.append(video_file)\n\n    print(\"\\n--- Embedding Deduplication Summary ---\")\n    print(f\"Total valid videos processed: {len(valid_videos)}\")\n    print(f\"Unique videos identified: {len(unique_videos)}\")\n    print(f\"Duplicate videos found: {len(duplicate_videos)}\")\n    print(f\"(Similarity Threshold: {similarity_threshold})\")\n\n    if unique_videos:\n        print(\"\\nUnique Videos:\")\n        for vid in unique_videos:\n            print(f\"- {os.path.basename(vid)}\")\n\n    if duplicate_videos:\n        print(\"\\nDuplicate Mappings (Duplicate -> Original):\")\n        for dup, orig in duplicate_videos.items():\n            print(f\"- {os.path.basename(dup)} -> {os.path.basename(orig)}\")\n import os
import logging
import cv2
from PIL import Image
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Import frame extraction from the other deduplication module
# In a larger project, this might be moved to a shared utils module
try:
    from deduplication import extract_sample_frames
except ImportError:
    logging.error("Could not import extract_sample_frames. Make sure deduplication.py is accessible.")
    # Define a dummy function if import fails, to allow script loading
    def extract_sample_frames(video_path, num_frames=5):
        logging.warning("Using dummy extract_sample_frames due to import error.")
        return []

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Model Loading ---
# Load the model only once when the module is imported for efficiency.
# Using a smaller CLIP model for faster local testing without GPU.
# For production, use a larger model and run on GPU.
MODEL_NAME = 'clip-vit-base-patch32'
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_embedding_model():
    global MODEL
    if MODEL is None:
        logger.info(f"Loading embedding model ({MODEL_NAME}) onto {DEVICE}...")
        start_load = time.time()
        try:
            MODEL = SentenceTransformer(MODEL_NAME, device=DEVICE)
            logger.info(f"Model loaded in {time.time() - start_load:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{MODEL_NAME}': {e}", exc_info=True)
            MODEL = None # Ensure model remains None if loading failed
    return MODEL

# --- Embedding Calculation ---
def calculate_video_embeddings(video_path, model, num_frames=5):
    """
    Calculates embeddings for sample frames of a video using a SentenceTransformer model.

    Args:
        video_path (str): Path to the video file.
        model (SentenceTransformer): The loaded embedding model.
        num_frames (int): Number of frames to sample.

    Returns:
        np.ndarray or None: A numpy array where each row is an embedding for a frame,
                            or None if embedding calculation failed.
    """
    if model is None:
        logger.error("Embedding model is not loaded. Cannot calculate embeddings.")
        return None

    embeddings = []
    frames = extract_sample_frames(video_path, num_frames)
    if not frames:
        logger.warning(f"No frames extracted from {video_path}, cannot calculate embeddings.")
        return None

    try:
        # Batch encode PIL images directly (supported by SentenceTransformer)
        logger.debug(f"Encoding {len(frames)} frames from {os.path.basename(video_path)}...")
        frame_embeddings = model.encode(frames, convert_to_numpy=True, show_progress_bar=False, device=DEVICE)
        logger.debug(f"Calculated {len(frame_embeddings)} embeddings for {os.path.basename(video_path)}")
        # Return as a single NumPy array (num_frames x embedding_dim)
        return frame_embeddings
    except Exception as e:
        logger.error(f"Error calculating embeddings for frames from {video_path}: {e}", exc_info=True)
        return None

# --- Duplicate Detection --- 
def find_duplicate_by_embedding(video_path, model, existing_embeddings_db, num_frames=5, threshold=0.95):
    """
    Checks if a video is likely a duplicate based on embedding similarity.

    Args:
        video_path (str): Path to the video file to check.
        model (SentenceTransformer): The loaded embedding model.
        existing_embeddings_db (dict): {video_id: np.ndarray_of_frame_embeddings}
        num_frames (int): Number of frames to sample.
        threshold (float): Minimum average cosine similarity to be considered a duplicate.

    Returns:
        tuple: (str, float) -> (path_of_duplicate, average_similarity) if duplicate found,
               otherwise None.
    """
    current_embeddings = calculate_video_embeddings(video_path, model, num_frames)
    if current_embeddings is None or current_embeddings.shape[0] == 0:
        logger.warning(f"Could not calculate embeddings for {video_path}. Skipping duplicate check.")
        return None

    max_avg_similarity = -1.0
    closest_match_id = None

    for existing_id, existing_embeddings in existing_embeddings_db.items():
        # Check if shapes match (same number of frames sampled)
        if existing_embeddings is None or current_embeddings.shape != existing_embeddings.shape:
             logger.debug(f"Embedding shape mismatch comparing {os.path.basename(video_path)} ({current_embeddings.shape}) and {os.path.basename(existing_id)} ({existing_embeddings.shape if existing_embeddings is not None else 'None'}). Skipping.")
             continue

        try:
            # Calculate cosine similarity between corresponding frame embeddings
            similarities = cosine_similarity(current_embeddings, existing_embeddings)
            # We want the diagonal elements, as we compare frame 1 to frame 1, frame 2 to frame 2, etc.
            pair_similarities = np.diag(similarities)
            
            if len(pair_similarities) > 0:
                average_similarity = np.mean(pair_similarities)
                logger.debug(f"Comparing {os.path.basename(video_path)} with {os.path.basename(existing_id)}: Avg similarity={average_similarity:.4f}")

                if average_similarity >= threshold:
                    if average_similarity > max_avg_similarity:
                        max_avg_similarity = average_similarity
                        closest_match_id = existing_id
                        logger.info(f"Found potential duplicate for {video_path}: {existing_id} with avg similarity {average_similarity:.4f}")
        except Exception as e:
            logger.error(f"Error calculating similarity between {video_path} and {existing_id}: {e}", exc_info=True)
            continue
        
    if closest_match_id is not None:
        return (closest_match_id, max_avg_similarity)

    return None

# --- Main function for testing ---
if __name__ == "__main__":
    # This test requires actual video files for meaningful results.
    # Dummy files won't work as frame extraction will fail.
    try:
        from ingestion import find_videos
    except ImportError:
        print("Error: Could not import 'find_videos' from ingestion.py.")
        exit()

    # Load the model first
    model = load_embedding_model()
    if model is None:
         print("Failed to load the embedding model. Cannot run deduplication.")
         exit()

    test_input_dir = "data/input_videos"
    processed_video_embeddings = {} # Store embeddings {path: np.array}
    unique_videos = []
    duplicate_videos = {} # Store mapping {duplicate_path: original_path}

    # --- NOTE --- 
    print("\n--- NOTE ---")
    print("Running EMBEDDING-based deduplication test.")
    print(f"This requires actual video files in '{test_input_dir}'.")
    print("Place some identical videos and some different ones.")
    print("Model used: ", MODEL_NAME)
    print("Device: ", DEVICE)
    print("------------\n")
    # Optionally create dummy *directory* if needed, but files must be real
    if not os.path.exists(test_input_dir):
        os.makedirs(test_input_dir)
        print(f"Created directory {test_input_dir}. Please add videos.")

    # 1. Find videos
    all_videos = find_videos(test_input_dir)
    valid_videos = [f for f in all_videos if os.path.exists(f) and os.path.getsize(f) > 0]

    if not valid_videos:
        print(f"No valid video files found in {test_input_dir}. Exiting.")
        exit()

    print(f"Found {len(valid_videos)} valid videos. Calculating embeddings and checking duplicates...")
    similarity_threshold = 0.95 # High threshold for likely duplicates

    # 2. Iterate and check duplicates
    for video_file in valid_videos:
        print(f"Processing: {os.path.basename(video_file)}")
        # Calculate embeddings for the current video
        current_embeddings = calculate_video_embeddings(video_file, model)
        
        if current_embeddings is None:
            print(f"  -> Result: ERROR processing (could not get embeddings)")
            continue # Skip this video if embeddings failed

        # Find duplicate by comparing with the existing DB
        duplicate_info = find_duplicate_by_embedding(video_file, model, processed_video_embeddings, threshold=similarity_threshold)

        if duplicate_info:
            original_id, similarity = duplicate_info
            print(f"  -> Result: DUPLICATE of {os.path.basename(original_id)} (Similarity: {similarity:.4f})")
            duplicate_videos[video_file] = original_id
        else:
            # Not a duplicate (or embeddings failed for comparison video), add to DB
            print(f"  -> Result: UNIQUE (added to known videos)")
            processed_video_embeddings[video_file] = current_embeddings
            unique_videos.append(video_file)

    print("\n--- Embedding Deduplication Summary ---")
    print(f"Total valid videos processed: {len(valid_videos)}")
    print(f"Unique videos identified: {len(unique_videos)}")
    print(f"Duplicate videos found: {len(duplicate_videos)}")
    print(f"(Similarity Threshold: {similarity_threshold})")

    if unique_videos:
        print("\nUnique Videos:")
        for vid in unique_videos:
            print(f"- {os.path.basename(vid)}")

    if duplicate_videos:
        print("\nDuplicate Mappings (Duplicate -> Original):")
        for dup, orig in duplicate_videos.items():
            print(f"- {os.path.basename(dup)} -> {os.path.basename(orig)}")