import ray
import os
import logging
import time
import json
import pandas as pd # For potentially easier handling of results
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch # For checking GPU availability

# Import functions from our modules
from ingestion import find_videos
from preprocessing import extract_metadata
# Import both placeholder and potentially real labeling function
import labeling # Import the whole module

# Deduplication methods
from deduplication import calculate_video_phashes as calculate_video_phashes_phash
# Embedding deduplication (optional import, handle gracefully if dependencies missing)
try:
    from deduplication_embeddings import (
        load_embedding_model, 
        calculate_video_embeddings, 
        find_duplicate_by_embedding
    )
    EMBEDDING_MODULE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Embedding deduplication module not available. pip install -r requirements.txt? Error: {e}")
    EMBEDDING_MODULE_AVAILABLE = False
    load_embedding_model = None
    calculate_video_embeddings = None
    find_duplicate_by_embedding = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# logger = logging.getLogger(__name__) # Use root logger configured in main

# --- Ray Task Definitions ---

# Deduplication Tasks (Unchanged)
@ray.remote
def process_video_for_phash(video_path, hash_size):
    """Ray task to calculate phashes for a single video."""
    task_logger = logging.getLogger(f"ray_task_phash_{os.path.basename(video_path)}") 
    task_logger.info(f"PHash task started (size={hash_size}) for: {os.path.basename(video_path)}")
    phashes = calculate_video_phashes_phash(video_path, hash_size=hash_size) 
    if phashes:
        task_logger.info(f"PHash task finished for: {os.path.basename(video_path)} - Success")
    else:
        task_logger.warning(f"PHash task finished for: {os.path.basename(video_path)} - Failed")
    return (video_path, phashes) if phashes else (video_path, None)

@ray.remote(num_cpus=1) 
def process_video_for_embedding(video_path, model_ref):
    """Ray task to calculate embeddings for a single video."""
    task_logger = logging.getLogger(f"ray_task_embed_{os.path.basename(video_path)}")
    task_logger.info(f"Embedding task started for: {os.path.basename(video_path)}")
    model = ray.get(model_ref)
    if model is None:
         task_logger.error(f"Embedding model not loaded in worker for {video_path}. Cannot proceed.")
         return (video_path, None)
    embeddings = calculate_video_embeddings(video_path, model)
    if embeddings is not None:
        task_logger.info(f"Embedding task finished for: {os.path.basename(video_path)} - Success ({embeddings.shape[0]} frames)")
    else:
        task_logger.warning(f"Embedding task finished for: {os.path.basename(video_path)} - Failed")
    return (video_path, embeddings) if embeddings is not None else (video_path, None)

# Preprocessing Task (Unchanged)
@ray.remote
def preprocess_video(video_path):
    """Ray task to extract technical metadata."""
    task_logger = logging.getLogger(f"ray_task_preprocess_{os.path.basename(video_path)}")
    task_logger.info(f"Preprocessing task started for: {os.path.basename(video_path)}")
    metadata = extract_metadata(video_path)
    if metadata:
        task_logger.info(f"Preprocessing task finished for: {os.path.basename(video_path)} - Success")
    else:
        task_logger.warning(f"Preprocessing task finished for: {os.path.basename(video_path)} - Failed")
    return metadata

# --- Labeling Task (Potentially using GPU) ---
# Check if GPU is available FOR RAY TASK SCHEDULING
# Note: This check on the driver doesn't guarantee Ray workers have GPU access,
# but helps configure the remote task request.
# Ray handles the actual scheduling to nodes with GPUs if requested.
NUM_GPUS_PER_LABELING_TASK = 1 if torch.cuda.is_available() else 0

@ray.remote(num_cpus=1, num_gpus=NUM_GPUS_PER_LABELING_TASK) # Request 1 CPU and potentially 1 GPU
def label_video_task(metadata, use_real_labels):
    """Ray task to apply AI labels (placeholder or real)."""
    if not metadata or not metadata.get('filepath'):
        logging.getLogger("ray_task_labeling").warning("Labeling task received invalid metadata.")
        return None
    video_basename = os.path.basename(metadata['filepath'])
    task_logger = logging.getLogger(f"ray_task_label_{video_basename}")
    task_logger.info(f"Labeling task started for: {video_basename} (Real labels: {use_real_labels})")

    # Select the labeling function based on the flag
    labeling_function = labeling.generate_labels # This now points to the real implementation
    
    # Inside the Ray task, especially if using GPU, ensure models are loaded.
    # Calling initialize_models() here attempts to load them within the worker process.
    # This relies on the global loading mechanism within labeling.py.
    # More robust: Use Ray Actors with GPU assignment for model serving.
    if use_real_labels:
        task_logger.info(f"Initializing models within Ray task for {video_basename}...")
        labeling.initialize_models() # Ensure models are loaded in this worker

    # Execute the chosen labeling function
    labeled_data = labeling_function(metadata['filepath'], metadata)
    
    if labeled_data:
        task_logger.info(f"Labeling task finished for: {video_basename} - Success")
    else:
         task_logger.warning(f"Labeling task finished for: {video_basename} - Failed")
    return labeled_data

# --- Main Pipeline Logic --- 
def run_local_pipeline(input_dir="data/input_videos", 
                       output_dir="data/pipeline_output",
                       dedup_method='phash', 
                       dedup_phash_size=8,  
                       dedup_phash_threshold=5,  
                       dedup_embedding_threshold=0.95,
                       num_frames_dedup=5, 
                       use_real_labels=False, # Flag for real vs placeholder labeling
                       num_cpus=None):
    """
    Runs the simplified local video processing pipeline using Ray.

    Args:
        input_dir (str): Directory containing input video files.
        output_dir (str): Directory to save the final labeled metadata.
        dedup_method (str): Deduplication method ('phash' or 'embedding').
        dedup_phash_size (int): Perceptual hash size (if method='phash').
        dedup_phash_threshold (int): Max Hamming distance (if method='phash').
        dedup_embedding_threshold (float): Min cosine similarity (if method='embedding').
        num_frames_dedup (int): Number of frames to sample for deduplication.
        use_real_labels (bool): If True, use real AI models for labeling (requires GPUs, more time).
                                  Defaults to False (uses placeholders).
        num_cpus (int, optional): Number of CPUs for Ray to use. Defaults to all available.
    """
    start_time = time.time()
    pipeline_logger = logging.getLogger("PipelineRunner")

    # --- Input Validation ---
    if dedup_method == 'embedding' and not EMBEDDING_MODULE_AVAILABLE:
        pipeline_logger.error("Embedding deduplication selected, but module/dependencies are missing.")
        return
    if dedup_method not in ['phash', 'embedding']:
        pipeline_logger.error(f"Invalid deduplication method: {dedup_method}.")
        return
    if use_real_labels and not torch.cuda.is_available():
        pipeline_logger.warning("Real labels requested but no CUDA GPU detected. Labeling will be VERY slow on CPU.")
    if use_real_labels:
         pipeline_logger.info("Using REAL AI models for labeling. This will take significantly longer and may require substantial resources (GPU recommended).")
    else:
         pipeline_logger.info("Using PLACEHOLDER functions for labeling.")

    pipeline_logger.info(f"Starting local pipeline with deduplication method: {dedup_method}")

    # --- Initialization ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        pipeline_logger.info(f"Created output directory: {output_dir}")

    # Initialize Ray
    # Consider initializing Ray with GPU information if appropriate
    # ray.init(num_cpus=num_cpus, num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0, ...)
    if not ray.is_initialized():
        gpu_count_str = f"GPUs={torch.cuda.device_count() if torch.cuda.is_available() else 0}" 
        pipeline_logger.info(f"Initializing Ray... (Using {'all' if num_cpus is None else num_cpus} CPUs, System {gpu_count_str}) ")
        ray.init(num_cpus=num_cpus, ignore_reinit_error=True,
                 logging_level=logging.WARNING) 
                 
    # Load embedding model if needed (driver side)
    model_ref = None
    if dedup_method == 'embedding':
        pipeline_logger.info("Loading embedding model for deduplication...")
        embedding_model = load_embedding_model() 
        if embedding_model is None:
             pipeline_logger.error("Failed to load embedding model. Cannot proceed.")
             if ray.is_initialized(): ray.shutdown()
             return
        model_ref = ray.put(embedding_model)
        pipeline_logger.info("Deduplication embedding model placed in Ray object store.")
        # Clear driver copy to save memory
        # del embedding_model

    # --- Stage 1: Ingestion ---
    pipeline_logger.info(f"Stage 1: Ingesting videos from {input_dir}")
    video_files = find_videos(input_dir)
    if not video_files:
        pipeline_logger.warning(f"No videos found in {input_dir}. Exiting.")
        if ray.is_initialized(): ray.shutdown()
        return
    pipeline_logger.info(f"Found {len(video_files)} video files.")
    valid_video_files = [f for f in video_files if os.path.exists(f) and os.path.getsize(f) > 0]
    if len(valid_video_files) < len(video_files):
         pipeline_logger.warning(f"Filtered out {len(video_files) - len(valid_video_files)} empty or non-existent files.")
    if not valid_video_files:
        pipeline_logger.warning("No valid (non-empty) video files found. Exiting.")
        if ray.is_initialized(): ray.shutdown()
        return

    # --- Stage 2: Deduplication --- 
    pipeline_logger.info(f"Stage 2: Running deduplication (Method: {dedup_method})...")
    dedup_results = []
    if dedup_method == 'phash':
        pipeline_logger.info(f"Calculating perceptual hashes (size={dedup_phash_size})...")
        dedup_futures = [process_video_for_phash.remote(f, dedup_phash_size) for f in valid_video_files]
        dedup_results = ray.get(dedup_futures)
    elif dedup_method == 'embedding':
        pipeline_logger.info("Calculating frame embeddings...")
        dedup_futures = [process_video_for_embedding.remote(f, model_ref) for f in valid_video_files]
        dedup_results = ray.get(dedup_futures)
    
    # --- Process Dedup Results ---
    existing_signatures_db = {}
    unique_video_paths = []
    duplicate_map = {}
    pipeline_logger.info("Checking for duplicates...")
    for video_path, signature in dedup_results:
        if signature is None:
            pipeline_logger.warning(f"Skipping duplicate check for {video_path} (signature calculation failed). Treating as unique.")
            unique_video_paths.append(video_path)
            continue
        is_duplicate = False
        best_match_id = None
        best_match_score = float('-inf') if dedup_method == 'embedding' else float('inf')
        for existing_id, existing_signature in existing_signatures_db.items():
            if dedup_method == 'phash':
                phashes = signature; existing_phashes = existing_signature
                if not existing_phashes or len(phashes) != len(existing_phashes):
                    pipeline_logger.debug(f"PHash length mismatch comparing {os.path.basename(video_path)} and {os.path.basename(existing_id)}. Skipping.")
                    continue
                try:
                    total_distance = 0; comparisons = 0
                    for i, (h1, h2) in enumerate(zip(phashes, existing_phashes)):
                        if h1 is None or h2 is None:
                            pipeline_logger.warning(f"Found None hash at index {i} comparing {video_path} and {existing_id}. Skipping pair.")
                            continue
                        total_distance += (h1 - h2); comparisons += 1
                    if comparisons == len(phashes) and comparisons > 0:
                        avg_distance = total_distance / comparisons
                        if avg_distance <= dedup_phash_threshold and avg_distance < best_match_score:
                             best_match_score = avg_distance; best_match_id = existing_id; is_duplicate = True
                except Exception as e:
                     pipeline_logger.error(f"Error comparing phashes between {video_path} and {existing_id}: {e}", exc_info=True); continue
            elif dedup_method == 'embedding':
                embeddings = signature; existing_embeddings = existing_signature
                if existing_embeddings is None or embeddings.shape != existing_embeddings.shape:
                    pipeline_logger.debug(f"Embedding shape mismatch comparing {os.path.basename(video_path)} and {os.path.basename(existing_id)}. Skipping.")
                    continue
                try:
                    similarities = cosine_similarity(embeddings, existing_embeddings); pair_similarities = np.diag(similarities)
                    if len(pair_similarities) > 0:
                        average_similarity = np.mean(pair_similarities)
                        if average_similarity >= dedup_embedding_threshold and average_similarity > best_match_score:
                             best_match_score = average_similarity; best_match_id = existing_id; is_duplicate = True
                except Exception as e:
                    pipeline_logger.error(f"Error calculating similarity between {video_path} and {existing_id}: {e}", exc_info=True); continue
        if is_duplicate:
            score_str = f"Dist: {best_match_score:.2f}" if dedup_method == 'phash' else f"Sim: {best_match_score:.4f}"
            pipeline_logger.info(f"Duplicate found: {os.path.basename(video_path)} -> {os.path.basename(best_match_id)} ({score_str})")
            duplicate_map[video_path] = best_match_id
        else:
            pipeline_logger.debug(f"Unique video: {os.path.basename(video_path)}")
            unique_video_paths.append(video_path)
            existing_signatures_db[video_path] = signature
    pipeline_logger.info(f"Deduplication finished. Found {len(unique_video_paths)} unique videos and {len(duplicate_map)} duplicates.")
    if not unique_video_paths:
        pipeline_logger.warning("No unique videos found to process further. Exiting.")
        if ray.is_initialized(): ray.shutdown(); return

    # --- Stage 3: Preprocessing (on unique videos) ---
    pipeline_logger.info(f"Stage 3: Preprocessing {len(unique_video_paths)} unique videos...")
    preprocess_futures = [preprocess_video.remote(p) for p in unique_video_paths]
    preprocessing_results = ray.get(preprocess_futures)
    valid_metadata_list = [meta for meta in preprocessing_results if meta is not None]
    pipeline_logger.info(f"Successfully preprocessed {len(valid_metadata_list)} videos.")
    if not valid_metadata_list:
        pipeline_logger.warning("Preprocessing failed for all videos. Exiting.")
        if ray.is_initialized(): ray.shutdown(); return

    # --- Stage 4: Labeling (on successfully preprocessed videos) ---
    pipeline_logger.info(f"Stage 4: Applying labels ({'REAL' if use_real_labels else 'PLACEHOLDER'}) to {len(valid_metadata_list)} videos...")
    # Use the updated label_video_task, passing the use_real_labels flag
    labeling_futures = [label_video_task.remote(meta, use_real_labels) for meta in valid_metadata_list]
    final_results = ray.get(labeling_futures)

    # Filter out None results (errors during labeling)
    successful_labeled_data = [data for data in final_results if data is not None]
    pipeline_logger.info(f"Successfully labeled {len(successful_labeled_data)} videos.")

    # --- Stage 5: Output ---
    pipeline_logger.info("Stage 5: Saving results...")
    if successful_labeled_data:
        output_file_path = os.path.join(output_dir, "final_labeled_metadata.jsonl")
        try:
            with open(output_file_path, 'w') as f:
                for record in successful_labeled_data:
                    f.write(json.dumps(record, default=lambda x: x.item() if isinstance(x, np.generic) else x) + '\n') # Handle numpy types
            pipeline_logger.info(f"Saved final labeled data for {len(successful_labeled_data)} videos to {output_file_path}")
        except IOError as e:
            pipeline_logger.error(f"Failed to save final output file: {e}")
        try:
            df = pd.json_normalize(successful_labeled_data, sep='_', errors='ignore') 
            csv_output_path = os.path.join(output_dir, "final_labeled_metadata.csv")
            df.to_csv(csv_output_path, index=False)
            pipeline_logger.info(f"Saved final labeled data as CSV to {csv_output_path}")
        except Exception as e:
            pipeline_logger.error(f"Failed to save results as CSV: {e}", exc_info=True)
    else:
        pipeline_logger.warning("No videos were successfully labeled.")
    dup_map_path = os.path.join(output_dir, "duplicate_map.json")
    try:
         with open(dup_map_path, 'w') as f:
             json.dump(duplicate_map, f, indent=4)
         pipeline_logger.info(f"Saved duplicate map ({len(duplicate_map)} entries) to {dup_map_path}")
    except IOError as e:
         pipeline_logger.error(f"Failed to save duplicate map: {e}")

    # --- Cleanup ---
    if ray.is_initialized():
        ray.shutdown()
    end_time = time.time()
    pipeline_logger.info(f"Pipeline finished in {end_time - start_time:.2f} seconds.")

# --- Entry Point ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the local video processing pipeline.")
    parser.add_argument("-i", "--input-dir", default="data/input_videos", help="Directory containing input video files.")
    parser.add_argument("-o", "--output-dir", default="data/pipeline_output", help="Directory to save the final labeled metadata.")
    
    # Deduplication args
    parser.add_argument("--dedup-method", default="phash", choices=["phash", "embedding"], help="Deduplication method.")
    parser.add_argument("--phash-size", type=int, default=8, help="Perceptual hash size for phash.")
    parser.add_argument("--phash-threshold", type=int, default=5, help="Max Hamming distance for phash.")
    parser.add_argument("--embedding-threshold", type=float, default=0.95, help="Min cosine similarity for embedding.")
    parser.add_argument("--num-frames-dedup", type=int, default=5, help="Number of frames for deduplication.")
    
    # Labeling args
    parser.add_argument("--use-real-labels", action="store_true", 
                        help="Use real AI models for labeling (slow, requires GPU & dependencies). Defaults to placeholders.")

    # Resource args
    parser.add_argument("--num-cpus", type=int, default=None, help="Number of CPUs for Ray.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")

    args = parser.parse_args()

    # Update logging level
    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level_int, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', force=True)

    INPUT_VIDEO_DIR = args.input_dir
    PIPELINE_OUTPUT_DIR = args.output_dir

    # Create dummy input dir/files check (unchanged, but note added)
    if not os.path.exists(INPUT_VIDEO_DIR): os.makedirs(INPUT_VIDEO_DIR)
    if not os.listdir(INPUT_VIDEO_DIR):
         dummy_path = os.path.join(INPUT_VIDEO_DIR, "pipeline_test_video.mp4")
         if not os.path.exists(dummy_path): 
             with open(dummy_path, 'w') as f: pass
             logging.info(f"Created empty dummy file for testing: {dummy_path}")
         print("\n--- NOTE ---")
         print(f"Input directory '{INPUT_VIDEO_DIR}' was empty or only contained the dummy file.")
         print("Pipeline will run, but results limited. Place actual video files for meaningful test.")
         if args.use_real_labels:
             print("Real labeling selected - requires non-empty, valid video files.")
         print("------------\n")

    # Run the pipeline with arguments
    run_local_pipeline(input_dir=INPUT_VIDEO_DIR, 
                       output_dir=PIPELINE_OUTPUT_DIR,
                       dedup_method=args.dedup_method,
                       dedup_phash_size=args.phash_size, 
                       dedup_phash_threshold=args.phash_threshold, 
                       dedup_embedding_threshold=args.embedding_threshold,
                       num_frames_dedup=args.num_frames_dedup,
                       use_real_labels=args.use_real_labels, # Pass the flag
                       num_cpus=args.num_cpus) 