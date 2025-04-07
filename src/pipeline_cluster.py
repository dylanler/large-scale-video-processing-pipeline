import ray
import os
import logging
import time
import json
import argparse
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse

# Ray Actor Pool for managing multiple actor instances
from ray.util.actor_pool import ActorPool

# Storage filesystem interaction (using pyarrow for example)
# Needs appropriate backend installed: pip install s3fs, gcsfs, etc.
import pyarrow.fs

# Import functions from local modules
# These might need adjustments if they rely heavily on local FS assumptions
from ingestion import find_videos # We'll replace its usage
from preprocessing import extract_metadata
# Labeling module contains model loading and real functions
import labeling 
# Deduplication modules
from deduplication import calculate_video_phashes as calculate_video_phashes_phash
try:
    from deduplication_embeddings import (
        load_embedding_model, 
        calculate_video_embeddings
    )
    EMBEDDING_MODULE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Embedding deduplication module not available: {e}")
    EMBEDDING_MODULE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# --- Filesystem Helper --- 
def get_pyarrow_fs(path: str) -> pyarrow.fs.FileSystem:
    """Initializes a pyarrow filesystem object based on the path prefix."""
    parsed_uri = urlparse(path)
    scheme = parsed_uri.scheme or "file"
    try:
        # TODO: Add configuration for credentials (e.g., environment variables, Ray secrets)
        fs, normalized_path = pyarrow.fs.FileSystem.from_uri(path)
        logging.info(f"Initialized {scheme} filesystem for path: {normalized_path}")
        return fs
    except Exception as e:
        logging.error(f"Failed to initialize filesystem for {path} (scheme: {scheme}): {e}")
        raise

def list_video_files_storage(input_path: str) -> List[str]:
    """Lists video files in a storage path (e.g., s3://..., gs://..., /local/...)."""
    pipeline_logger = logging.getLogger("PipelineRunner")
    pipeline_logger.info(f"Listing video files in: {input_path}")
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv') 
    try:
        fs = get_pyarrow_fs(input_path)
        selector = pyarrow.fs.FileSelector(input_path, recursive=True)
        files_info = fs.get_file_info(selector)
        
        video_files = []
        count = 0
        for file_info in files_info:
            if not file_info.is_file: continue
            path = file_info.path
            if path.lower().endswith(video_extensions):
                # Construct full URI if needed (fs might return relative paths)
                full_uri = f"{urlparse(input_path).scheme}://{path}" if urlparse(input_path).scheme else path
                video_files.append(full_uri)
                count += 1
                if count % 10000 == 0:
                     pipeline_logger.info(f"Found {count} video files so far...")
        
        pipeline_logger.info(f"Found {len(video_files)} video files in total.")
        # TODO: Check file sizes here if possible/needed using file_info.size
        # Filter out zero-byte files early
        # valid_video_files = [f for f in video_files if fs.get_file_info(f).size > 0] 
        return video_files
    except Exception as e:
        pipeline_logger.error(f"Error listing files in {input_path}: {e}", exc_info=True)
        return []

# --- Ray Actors for Model Serving --- 

@ray.remote(num_cpus=1, num_gpus=1) # Assign 1 GPU to each labeling actor
class LabelerActor:
    """A Ray actor that holds AI models for labeling and performs inference."""
    def __init__(self, model_keys_to_load: List[str]):
        self.logger = logging.getLogger(f"LabelerActor_{os.getpid()}")
        self.logger.info(f"Initializing on device: {labeling.DEVICE}")
        self.model_keys = model_keys_to_load
        # Load specific models this actor is responsible for
        labeling.initialize_models() # This loads all models defined in labeling.py
        self.logger.info(f"Models ({', '.join(self.model_keys)}) loaded.")
        # Store references to the specific functions needed
        self._func_map = {
            "caption": labeling.generate_video_caption_real,
            "objects": labeling.detect_objects_real,
            "actions": labeling.recognize_actions_real,
            "scene": labeling.classify_scene_real,
            "audio": labeling.analyze_audio_real,
            "additional": labeling.generate_additional_metadata_real,
        }

    def label(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Performs all configured labeling steps for a single video metadata dict."""
        video_path = metadata.get('filepath')
        if not video_path:
            self.logger.error("Received metadata without filepath.")
            return None
            
        self.logger.info(f"Labeling started for: {os.path.basename(video_path)}")
        start_time = time.time()
        
        labeled_results = {}
        object_results = None # Needed for additional metadata
        
        # Execute labeling functions sequentially within the actor
        # This could be parallelized too if models allow concurrent GPU use
        try:
            if "caption" in self.model_keys:
                labeled_results["ai_caption"] = self._func_map["caption"](video_path)
            if "objects" in self.model_keys:
                 object_results = self._func_map["objects"](video_path)
                 labeled_results["ai_objects"] = object_results
            if "actions" in self.model_keys:
                 labeled_results["ai_actions"] = self._func_map["actions"](video_path)
            if "scene" in self.model_keys:
                 scene_info = self._func_map["scene"](video_path)
                 labeled_results["ai_scene_type"] = scene_info.get("scene_type")
                 labeled_results["ai_environment"] = scene_info.get("environment")
            if "audio" in self.model_keys:
                 audio_info = self._func_map["audio"](video_path)
                 labeled_results["ai_has_speech"] = audio_info.get("has_speech")
                 labeled_results["ai_language"] = audio_info.get("language")
                 labeled_results["ai_sound_events"] = audio_info.get("sound_events")
                 labeled_results["ai_transcription_snippet"] = audio_info.get("transcription_snippet")
            if "additional" in self.model_keys:
                 additional_meta = self._func_map["additional"](video_path, object_results)
                 labeled_results["ai_visual_style"] = additional_meta.get("visual_style")
                 labeled_results["ai_mood_estimation"] = additional_meta.get("mood_estimation")
                 labeled_results["ai_camera_motion"] = additional_meta.get("camera_motion")
                 labeled_results["ai_estimated_people_count"] = additional_meta.get("estimated_people_count")

            labeled_results["ai_labeling_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            
            # Combine with original metadata
            final_metadata = metadata.copy()
            final_metadata.update(labeled_results)
            
            duration = time.time() - start_time
            self.logger.info(f"Labeling finished for: {os.path.basename(video_path)} in {duration:.2f}s")
            return final_metadata
            
        except Exception as e:
             self.logger.error(f"Error during labeling for {video_path}: {e}", exc_info=True)
             return None # Indicate failure

# --- Ray Tasks for CPU-bound work --- 

@ray.remote(num_cpus=1)
def process_video_for_phash_cluster(video_uri: str, hash_size: int) -> Tuple[str, Any]:
    """Ray task to calculate phashes for a single video URI."""
    # Needs access to the video file - assumes worker can read the URI (e.g., via mounted storage or fsspec)
    task_logger = logging.getLogger(f"ray_task_phash_{os.path.basename(video_uri)}")
    task_logger.info(f"PHash task started (size={hash_size}) for: {video_uri}")
    # TODO: Handle potential download/streaming if URI is remote & not auto-handled
    phashes = calculate_video_phashes_phash(video_uri, hash_size=hash_size)
    # ... (rest is same as local version) ...
    return (video_uri, phashes) if phashes else (video_uri, None)

@ray.remote(num_cpus=1)
def process_video_for_embedding_cluster(video_uri: str, model_ref: Any) -> Tuple[str, Any]:
    """Ray task to calculate embeddings for a single video URI."""
    task_logger = logging.getLogger(f"ray_task_embed_{os.path.basename(video_uri)}")
    task_logger.info(f"Embedding task started for: {video_uri}")
    model = ray.get(model_ref) # Get shared model
    if model is None: return (video_uri, None)
    # TODO: Handle download/streaming
    embeddings = calculate_video_embeddings(video_uri, model)
    # ... (rest is same as local version) ...
    return (video_uri, embeddings) if embeddings is not None else (video_uri, None)

@ray.remote(num_cpus=1)
def preprocess_video_cluster(video_uri: str) -> Dict[str, Any]:
    """Ray task to extract technical metadata from a video URI."""
    task_logger = logging.getLogger(f"ray_task_preprocess_{os.path.basename(video_uri)}")
    task_logger.info(f"Preprocessing task started for: {video_uri}")
    # TODO: Handle download/streaming
    metadata = extract_metadata(video_uri)
    # Important: Update filepath in metadata to be the URI for consistency
    if metadata: metadata['filepath'] = video_uri 
    # ... (rest is same as local version) ...
    return metadata

# --- Main Pipeline Logic --- 
def run_cluster_pipeline(args: argparse.Namespace):
    """
    Runs the video processing pipeline on a Ray cluster.
    """
    start_time = time.time()
    pipeline_logger = logging.getLogger("PipelineRunner")
    pipeline_logger.info(f"Starting CLUSTER pipeline with args: {args}")

    # --- Input Validation ---
    if args.dedup_method == 'embedding' and not EMBEDDING_MODULE_AVAILABLE:
        pipeline_logger.error("Embedding deduplication selected, but module/dependencies missing.")
        return
    # ... (other validation as needed) ...

    # --- Ray Connection ---
    pipeline_logger.info(f"Connecting to Ray cluster at address: '{args.ray_address}'")
    # Requires a running Ray cluster. Address might be "auto" or specific like "ray://<head_node_ip>:10001"
    ray.init(address=args.ray_address, ignore_reinit_error=True, logging_level=logging.WARNING)
    pipeline_logger.info(f"Ray cluster resources: {ray.cluster_resources()}")
                 
    # --- Load & Share Embedding Model (if needed for dedup) ---
    model_ref = None
    if args.dedup_method == 'embedding':
        pipeline_logger.info("Loading embedding model for deduplication (driver side)...")
        # Load on driver first to put in object store
        embedding_model = load_embedding_model() 
        if embedding_model is None: raise RuntimeError("Failed to load embedding model.")
        model_ref = ray.put(embedding_model)
        pipeline_logger.info("Deduplication embedding model placed in Ray object store.")
        del embedding_model # Free driver memory

    # --- Stage 1: Ingestion --- 
    # Replace local find_videos with listing from storage
    video_uris = list_video_files_storage(args.input_path)
    if not video_uris:
        pipeline_logger.warning(f"No video files found in {args.input_path}. Exiting.")
        if ray.is_initialized(): ray.shutdown(); return
    pipeline_logger.info(f"Found {len(video_uris)} video files for processing.")
    # Optional: Filter URIs based on size or other criteria if needed
    valid_video_uris = video_uris # Assume all are valid for now

    # --- Stage 2: Deduplication --- 
    pipeline_logger.info(f"Stage 2: Calculating signatures for deduplication (Method: {args.dedup_method})...")
    dedup_futures = []
    if args.dedup_method == 'phash':
        dedup_futures = [process_video_for_phash_cluster.remote(uri, args.phash_size) for uri in valid_video_uris]
    elif args.dedup_method == 'embedding':
        dedup_futures = [process_video_for_embedding_cluster.remote(uri, model_ref) for uri in valid_video_uris]
        
    # --- Process Dedup Signatures & Identify Unique URIs --- 
    # !!! Placeholder for Scalable Deduplication !!!
    # In a real cluster scenario, this step is complex and involves external systems.
    pipeline_logger.info("Gathering deduplication signatures...")
    dedup_results = ray.get(dedup_futures) # List of (uri, signature)
    
    pipeline_logger.warning("--- Deduplication Comparison Placeholder --- ")
    pipeline_logger.warning("This step requires integration with a scalable Vector DB (Milvus, Weaviate, Pinecone) or distributed Faiss.")
    # 1. Write `dedup_results` (signatures) to the Vector DB or a staging area.
    # 2. Perform large-scale nearest neighbor search in the Vector DB to find duplicates.
    # 3. Obtain the list of unique video URIs based on the Vector DB results.
    # For this script, we'll proceed assuming all non-failed signatures correspond to unique videos.
    unique_video_uris = [uri for uri, sig in dedup_results if sig is not None]
    failed_dedup_count = len(valid_video_uris) - len(unique_video_uris)
    duplicate_map = {} # Map would be generated by external system
    pipeline_logger.warning(f"Proceeding with {len(unique_video_uris)} videos assuming they are unique (after filtering {failed_dedup_count} signature failures).")
    pipeline_logger.warning("--- End Placeholder --- ")
    
    if not unique_video_uris:
        pipeline_logger.warning("No unique videos identified after deduplication placeholder. Exiting.")
        if ray.is_initialized(): ray.shutdown(); return

    # --- Stage 3: Preprocessing (on unique videos) ---
    pipeline_logger.info(f"Stage 3: Preprocessing {len(unique_video_uris)} unique videos...")
    preprocess_futures = [preprocess_video_cluster.remote(uri) for uri in unique_video_uris]
    preprocessing_results = ray.get(preprocess_futures)
    valid_metadata_list = [meta for meta in preprocessing_results if meta is not None]
    failed_preprocess_count = len(unique_video_uris) - len(valid_metadata_list)
    pipeline_logger.info(f"Successfully preprocessed {len(valid_metadata_list)} videos ({failed_preprocess_count} failures).")
    if not valid_metadata_list:
        pipeline_logger.warning("Preprocessing failed for all videos. Exiting.")
        if ray.is_initialized(): ray.shutdown(); return

    # --- Stage 4: Labeling (using Actor Pool) ---
    pipeline_logger.info(f"Stage 4: Applying REAL labels to {len(valid_metadata_list)} videos using Actor Pool...")
    # Define which labels we want (all in this case)
    all_label_keys = ["caption", "objects", "actions", "scene", "audio", "additional"]
    
    # Create a pool of LabelerActors on GPUs
    # Adjust pool size based on available GPUs and desired parallelism
    num_labeling_actors = args.num_labeling_workers # Use CLI arg
    pipeline_logger.info(f"Creating ActorPool with {num_labeling_actors} LabelerActor instances (requesting 1 GPU each)...")
    # Ensure actor class requests GPUs correctly
    actor_cls = LabelerActor.options(num_cpus=1, num_gpus=1) 
    labeler_pool = ActorPool([actor_cls.remote(all_label_keys) for _ in range(num_labeling_actors)])
    pipeline_logger.info("Submitting labeling tasks to ActorPool...")
    
    # Submit tasks using map_unordered for better load balancing
    results_iterator = labeler_pool.map_unordered(lambda actor, meta: actor.label.remote(meta), valid_metadata_list)
    
    successful_labeled_data = []
    labeling_failures = 0
    processed_count = 0
    start_labeling_stage = time.time()
    for result in results_iterator:
        processed_count += 1
        if result is not None:
            successful_labeled_data.append(result)
        else:
            labeling_failures += 1
        if processed_count % 1000 == 0:
            elapsed = time.time() - start_labeling_stage
            rate = processed_count / elapsed if elapsed > 0 else 0
            pipeline_logger.info(f"Labeling progress: {processed_count}/{len(valid_metadata_list)} processed ({labeling_failures} failures). Rate: {rate:.2f} videos/sec")

    pipeline_logger.info(f"Labeling stage finished. Successfully labeled {len(successful_labeled_data)} videos ({labeling_failures} failures).")

    # --- Stage 5: Output --- 
    pipeline_logger.info(f"Stage 5: Saving results to {args.output_path}...")
    if successful_labeled_data:
        # Example: Save to a JSON Lines file in object storage
        output_file = os.path.join(args.output_path, "final_labeled_metadata.jsonl")
        pipeline_logger.info(f"Writing {len(successful_labeled_data)} records to {output_file}")
        try:
            fs = get_pyarrow_fs(output_file)
            # Ensure output directory exists (needed for some filesystems like local/HDFS)
            output_dir = os.path.dirname(output_file)
            if urlparse(output_dir).scheme in ["", "file", "hdfs"]:
                 fs.create_dir(output_dir, recursive=True)
                 
            with fs.open_output_stream(output_file) as f:
                for record in successful_labeled_data:
                    # Add default handler for numpy types during JSON dump
                    f.write((json.dumps(record, default=lambda x: x.item() if isinstance(x, np.generic) else x) + '\n').encode('utf-8'))
            pipeline_logger.info(f"Successfully saved final labeled data.")
        except Exception as e:
            pipeline_logger.error(f"Failed to save final output to {output_file}: {e}", exc_info=True)
    else:
        pipeline_logger.warning("No videos were successfully labeled.")

    # --- Cleanup ---
    # No need to save duplicate map as it's handled externally
    if ray.is_initialized():
        ray.shutdown()
    end_time = time.time()
    pipeline_logger.info(f"Cluster pipeline finished in {end_time - start_time:.2f} seconds.")

# --- Entry Point --- 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Video Processing Pipeline on a Ray Cluster.")
    
    # Cluster Args
    parser.add_argument("--ray-address", default="auto", help="Address of the Ray cluster ('auto' or ray://<head_ip>:10001).")
    
    # Path Args (Use Storage URIs: s3://..., gs://..., hdfs://..., /mnt/nfs/...)
    parser.add_argument("-i", "--input-path", required=True, help="Input path prefix in storage (e.g., s3://bucket/videos/).")
    parser.add_argument("-o", "--output-path", required=True, help="Output path prefix in storage (e.g., s3://bucket/results/).")
    
    # Deduplication Args
    parser.add_argument("--dedup-method", default="embedding", choices=["phash", "embedding"], help="Deduplication signature calculation method.")
    parser.add_argument("--phash-size", type=int, default=8, help="Perceptual hash size for phash.")
    # Thresholds for comparison are NOT used here as comparison is external
    # parser.add_argument("--phash-threshold", type=int, default=5, help="Max Hamming distance for phash.")
    # parser.add_argument("--embedding-threshold", type=float, default=0.95, help="Min cosine similarity for embedding.")
    parser.add_argument("--num-frames-dedup", type=int, default=5, help="Number of frames for deduplication signature.")

    # Labeling Args
    parser.add_argument("--num-labeling-workers", type=int, default=100, 
                        help="Number of LabelerActor instances to create in the pool (should approx match available GPUs).")
    # Real labels are assumed for cluster pipeline
    # parser.add_argument("--use-real-labels", action="store_true", help="Use real AI models for labeling.")

    # Resource Args (Optional overrides, primarily handled by Ray cluster config)
    # parser.add_argument("--num-cpus-per-task", ...)
    # parser.add_argument("--num-gpus-per-actor", ...)

    # Logging
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")

    args = parser.parse_args()

    # --- Setup Logging ---
    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level_int, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', force=True)
    logging.getLogger("pyarrow").setLevel(logging.WARNING) # Reduce pyarrow verbosity
    logging.getLogger("fsspec").setLevel(logging.WARNING) # Reduce fsspec verbosity

    # --- Run Pipeline ---
    run_cluster_pipeline(args) 