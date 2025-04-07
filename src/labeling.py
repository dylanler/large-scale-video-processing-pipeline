import os
import logging
import random
import time
import json
import cv2
from PIL import Image
import numpy as np
import torch
import torchaudio
import librosa
import scipy.signal
from transformers import (
    pipeline, 
    BlipProcessor, BlipForConditionalGeneration, 
    VideoMAEImageProcessor, VideoMAEForVideoClassification, 
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2FeatureExtractor, WavLMForAudioFrameClassification,
    AutoImageProcessor, AutoModelForObjectDetection # For object detection
)
from torchvision.transforms.functional import to_pil_image
from sentence_transformers import SentenceTransformer # Reusing for scene/style/mood

# Import frame extraction (adjust path if needed)
try:
    from deduplication import extract_sample_frames
except ImportError:
    logging.error("Could not import extract_sample_frames.")
    def extract_sample_frames(video_path, num_frames=5): return []

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Model Loading & Device Setup ---
# Load models globally to avoid reloading in each function call within the same process.
# In Ray tasks, this might happen once per worker process.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"Labeling using device: {DEVICE}")

MODELS = {}
PROCESSORS = {}

def load_model(model_key, model_class, processor_class=None, model_name=None, task=None):
    """Helper to load models and processors."""
    if model_key not in MODELS:
        logger.info(f"Loading model for {model_key} ({model_name or task})...")
        start_time = time.time()
        try:
            if task:
                # Use pipeline for simplicity if applicable
                MODELS[model_key] = pipeline(task, model=model_name, device=DEVICE)
            else:
                if processor_class:
                    PROCESSORS[model_key] = processor_class.from_pretrained(model_name)
                MODELS[model_key] = model_class.from_pretrained(model_name).to(DEVICE)
                MODELS[model_key].eval() # Set to evaluation mode
            logger.info(f"Loaded {model_key} in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}", exc_info=True)
            MODELS[model_key] = None
            if model_key in PROCESSORS: PROCESSORS[model_key] = None

# --- Model Keys and Names ---
CAPTION_KEY = 'caption'
CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
OBJECT_DET_KEY = 'object_detection'
OBJECT_DET_MODEL = "facebook/detr-resnet-50"
ACTION_REC_KEY = 'action_recognition'
ACTION_REC_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics400"
AUDIO_TRANS_KEY = 'audio_transcription'
AUDIO_TRANS_MODEL = "openai/whisper-base" # Use base for faster local test
AUDIO_EVENT_KEY = 'audio_event'
AUDIO_EVENT_MODEL = "microsoft/wavlm-base-plus-sv" # Example sound classification
# Reusing CLIP for Scene/Style/Mood similarity
CLIP_KEY = 'clip'
CLIP_MODEL = "openai/clip-vit-base-patch32"

# Pre-load models (can take time on first run)
def initialize_models():
    logger.info("Initializing AI labeling models...")
    # Captioning
    load_model(CAPTION_KEY, BlipForConditionalGeneration, BlipProcessor, CAPTION_MODEL)
    # Object Detection
    load_model(OBJECT_DET_KEY, AutoModelForObjectDetection, AutoImageProcessor, OBJECT_DET_MODEL)
    # Action Recognition
    load_model(ACTION_REC_KEY, VideoMAEForVideoClassification, VideoMAEImageProcessor, ACTION_REC_MODEL)
    # Audio Transcription
    load_model(AUDIO_TRANS_KEY, WhisperForConditionalGeneration, WhisperProcessor, AUDIO_TRANS_MODEL)
    # Audio Event Classification
    load_model(AUDIO_EVENT_KEY, WavLMForAudioFrameClassification, Wav2Vec2FeatureExtractor, AUDIO_EVENT_MODEL)
    # CLIP (via SentenceTransformer)
    global MODELS # Need to modify global MODELS
    if CLIP_KEY not in MODELS:
        logger.info(f"Loading model for {CLIP_KEY} ({CLIP_MODEL})...")
        start_time = time.time()
        try:
            MODELS[CLIP_KEY] = SentenceTransformer(CLIP_MODEL, device=DEVICE)
            logger.info(f"Loaded {CLIP_KEY} in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load model {CLIP_KEY}: {e}", exc_info=True)
            MODELS[CLIP_KEY] = None
    logger.info("Model initialization complete.")

# --- Helper Functions --- 
def extract_video_frames_for_action(video_path, num_frames=16, target_fps=30):
    """Extract frames specifically for video models like VideoMAE."""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return frames
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0: original_fps = 30 # Assume default if FPS read fails

        if total_frames <= 0: return frames

        # Calculate indices to sample `num_frames` evenly across the video duration
        duration = total_frames / original_fps
        if duration == 0: return frames
        
        # Sample based on time rather than frame count directly for consistency
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(i))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                 logger.warning(f"Could not read frame index {i} for action rec from {video_path}")
        cap.release()
    except Exception as e:
        logger.error(f"Error extracting frames for action rec from {video_path}: {e}", exc_info=True)
        if 'cap' in locals() and cap.isOpened(): cap.release()
    # Return list of numpy arrays (H, W, C)
    return frames if len(frames) == num_frames else [] # Ensure correct number of frames

def extract_audio(video_path, target_sr=16000):
    """Extracts audio waveform and sample rate using ffmpeg/torchaudio."""
    try:
        waveform, sr = torchaudio.load(video_path)
        # Resample if necessary
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
            sr = target_sr
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform.squeeze().numpy(), sr # Return numpy array for librosa/transformers
    except Exception as e:
        logger.error(f"Failed to load or resample audio from {video_path}: {e}")
        return None, None

# --- Real AI Labeling Functions ---

def generate_video_caption_real(video_path, num_frames=1):
    """Generates caption using BLIP model on sample frame(s)."""
    model = MODELS.get(CAPTION_KEY)
    processor = PROCESSORS.get(CAPTION_KEY)
    if not model or not processor:
        logger.warning("Caption model/processor not loaded.")
        return "[Captioning unavailable]"

    frames = extract_sample_frames(video_path, num_frames=num_frames) # Get PIL images
    if not frames:
        return "[Captioning failed: No frames]"
    
    try:
        # For simplicity, caption the middle frame if num_frames > 1
        frame_to_caption = frames[len(frames) // 2]
        
        inputs = processor(images=frame_to_caption, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50, num_beams=4, early_stopping=True)
        caption = processor.decode(out[0], skip_special_tokens=True)
        logger.debug(f"Generated caption for {os.path.basename(video_path)}: {caption}")
        return caption
    except Exception as e:
        logger.error(f"Error during caption generation for {video_path}: {e}", exc_info=True)
        return "[Captioning error]"

def detect_objects_real(video_path, num_frames=1, conf_threshold=0.7):
    """Detects objects using DETR model on sample frame(s)."""
    model = MODELS.get(OBJECT_DET_KEY)
    processor = PROCESSORS.get(OBJECT_DET_KEY)
    if not model or not processor:
        logger.warning("Object detection model/processor not loaded.")
        return {} # Return empty dict

    frames = extract_sample_frames(video_path, num_frames=num_frames)
    if not frames:
        return {} 
        
    detected_objects = {}
    try:
        # Process middle frame for simplicity
        frame_to_detect = frames[len(frames) // 2]
        inputs = processor(images=frame_to_detect, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Convert outputs to COCO API format (faster R-CNN, DETR output formats differ)
        target_sizes = torch.tensor([frame_to_detect.size[::-1]], device=DEVICE)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=conf_threshold)[0]
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = model.config.id2label[label.item()]
            confidence = round(score.item(), 2)
            # Aggregate counts or store with confidence
            if label_name not in detected_objects or detected_objects[label_name] < confidence:
                 detected_objects[label_name] = confidence
        logger.debug(f"Detected objects for {os.path.basename(video_path)}: {detected_objects}")
        
    except Exception as e:
        logger.error(f"Error during object detection for {video_path}: {e}", exc_info=True)
        return {"[Object detection error]": 0.0}
        
    return detected_objects

def recognize_actions_real(video_path, num_frames=16):
    """Recognizes actions using VideoMAE model."""
    model = MODELS.get(ACTION_REC_KEY)
    processor = PROCESSORS.get(ACTION_REC_KEY)
    if not model or not processor:
        logger.warning("Action recognition model/processor not loaded.")
        return ["[Action recognition unavailable]"]

    frames = extract_video_frames_for_action(video_path, num_frames=num_frames)
    if not frames:
        return ["[Action recognition failed: Not enough frames]"]
        
    try:
        inputs = processor(frames, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Get top-k predicted actions
        top_k = 5
        predicted_class_ids = logits.topk(k=top_k, dim=-1).indices.squeeze().tolist()
        if not isinstance(predicted_class_ids, list): predicted_class_ids = [predicted_class_ids]
            
        actions = [model.config.id2label[class_id] for class_id in predicted_class_ids]
        logger.debug(f"Recognized actions for {os.path.basename(video_path)} (top {top_k}): {actions}")
        # Return top 1 or top N based on requirements
        return actions[:1] if actions else ["[Action recognition failed]"]
    except Exception as e:
        logger.error(f"Error during action recognition for {video_path}: {e}", exc_info=True)
        return ["[Action recognition error]"]

def classify_scene_real(video_path, num_frames=1):
    """Classifies scene using CLIP similarity to predefined prompts."""
    model = MODELS.get(CLIP_KEY) # SentenceTransformer CLIP model
    if not model:
        logger.warning("CLIP model not loaded for scene classification.")
        return {"scene_type": "[Scene classification unavailable]", "environment": None}

    frames = extract_sample_frames(video_path, num_frames=num_frames)
    if not frames:
        return {"scene_type": "[Scene classification failed: No frames]", "environment": None}

    # Define scene and environment prompts
    scene_prompts = ["indoors", "outdoors", "cityscape", "landscape", "beach", "forest", "office", "kitchen", "street view", "aerial view"]
    env_prompts = ["daytime", "nighttime", "urban setting", "natural setting"]
    
    try:
        # Encode frame (middle one)
        frame_to_classify = frames[len(frames) // 2]
        frame_embedding = model.encode([frame_to_classify], convert_to_numpy=True, show_progress_bar=False)[0]
        
        # Encode text prompts
        scene_embeddings = model.encode(scene_prompts, convert_to_numpy=True, show_progress_bar=False)
        env_embeddings = model.encode(env_prompts, convert_to_numpy=True, show_progress_bar=False)
        
        # Calculate similarities
        scene_similarities = cosine_similarity([frame_embedding], scene_embeddings)[0]
        env_similarities = cosine_similarity([frame_embedding], env_embeddings)[0]
        
        # Get best match
        best_scene_idx = np.argmax(scene_similarities)
        best_env_idx = np.argmax(env_similarities)
        
        scene_type = scene_prompts[best_scene_idx]
        environment = env_prompts[best_env_idx]
        
        logger.debug(f"Classified scene for {os.path.basename(video_path)}: {scene_type} ({scene_similarities[best_scene_idx]:.2f}), {environment} ({env_similarities[best_env_idx]:.2f})")
        return {"scene_type": scene_type, "environment": environment}
        
    except Exception as e:
        logger.error(f"Error during scene classification for {video_path}: {e}", exc_info=True)
        return {"scene_type": "[Scene classification error]", "environment": None}

def analyze_audio_real(video_path, target_sr=16000):
    """Analyzes audio using Whisper and WavLM."""
    whisper_model = MODELS.get(AUDIO_TRANS_KEY)
    whisper_processor = PROCESSORS.get(AUDIO_TRANS_KEY)
    event_model = MODELS.get(AUDIO_EVENT_KEY)
    event_processor = PROCESSORS.get(AUDIO_EVENT_KEY)
    
    results = {
        "has_speech": False,
        "language": None,
        "transcription_snippet": None,
        "sound_events": ["[Audio analysis unavailable]"]
    }

    waveform, sr = extract_audio(video_path, target_sr)
    if waveform is None or sr is None:
        results["sound_events"] = ["[Audio analysis failed: Cannot load audio]"]
        return results

    # --- Transcription & Language ID (Whisper) ---
    if whisper_model and whisper_processor:
        try:
            inputs = whisper_processor(waveform, sampling_rate=sr, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                # Generate token ids
                predicted_ids = whisper_model.generate(inputs.input_features, max_length=100) # Limit snippet length
            # Decode token ids to text
            transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            
            if transcription:
                results["has_speech"] = True
                results["transcription_snippet"] = transcription
                # Simple language detection (Whisper often includes language token)
                # More robustly: use language identification features if available
                lang_token_id = predicted_ids[0][1].item() # Often the second token
                lang = whisper_processor.tokenizer.decode(lang_token_id) 
                if lang.startswith("<") and lang.endswith(">"): # e.g., <|en|>
                    results["language"] = lang[2:-2]
                logger.debug(f"Audio transcription snippet for {os.path.basename(video_path)} ({results['language']}): {transcription}")
            else:
                 logger.debug(f"No speech detected by Whisper for {os.path.basename(video_path)}.")
                 
        except Exception as e:
            logger.error(f"Error during audio transcription for {video_path}: {e}", exc_info=True)
            results["transcription_snippet"] = "[Transcription error]"
    else:
        logger.warning("Whisper model/processor not loaded.")

    # --- Sound Event Classification (WavLM example) ---
    if event_model and event_processor:
        try:
             inputs = event_processor(waveform, sampling_rate=sr, return_tensors="pt").to(DEVICE)
             with torch.no_grad():
                 outputs = event_model(**inputs)
                 logits = outputs.logits
                 
             # Get top N classes for the whole clip (or segments)
             # This WavLM model might be frame-level, need aggregation
             # Simple mean pooling of logits or max probability over time
             probabilities = torch.softmax(logits.mean(dim=1), dim=-1) # Example: mean pool
             top_prob, top_indices = torch.topk(probabilities.squeeze(), k=5)
             
             detected_events = []
             for prob, idx in zip(top_prob.tolist(), top_indices.tolist()):
                 if prob > 0.1: # Confidence threshold
                     detected_events.append(f"{event_model.config.id2label[idx]} ({prob:.2f})")
             
             if detected_events:
                 results["sound_events"] = detected_events
                 logger.debug(f"Detected audio events for {os.path.basename(video_path)}: {detected_events}")
             else:
                 results["sound_events"] = ["[No significant audio events detected]"]
                 
        except Exception as e:
            logger.error(f"Error during audio event detection for {video_path}: {e}", exc_info=True)
            results["sound_events"] = ["[Audio event detection error]"]
    else:
        logger.warning("Audio event model/processor not loaded.")

    return results

def generate_additional_metadata_real(video_path, object_results):
     """Generates other metadata based on object detection and CLIP."""
     clip_model = MODELS.get(CLIP_KEY)
     if not clip_model:
          logger.warning("CLIP model not loaded for additional metadata.")
          # Return defaults or placeholders
          return {
              "visual_style": "[Unavailable]",
              "mood_estimation": "[Unavailable]",
              "camera_motion": "[Not implemented]",
              "estimated_people_count": "[Unavailable]"
          }
          
     # --- People Count --- 
     people_count_str = "0"
     if object_results and isinstance(object_results, dict):
         person_conf = object_results.get("person", 0.0)
         if person_conf > 0:
             people_count_str = "1" # Simple estimate, could be improved
             # TODO: Could try to count distinct boxes over time if needed

     # --- Style/Mood (using CLIP similarity to prompts) ---
     visual_style = "[Unavailable]"
     mood_estimation = "[Unavailable]"
     frames = extract_sample_frames(video_path, num_frames=1)
     if frames:
         frame_to_classify = frames[0]
         try:
             style_prompts = ["cinematic style", "amateur footage", "animation style", "vlog style", "documentary style", "black and white film", "monochrome colors", "vibrant colors"]
             mood_prompts = ["calm mood", "energetic mood", "dramatic mood", "neutral mood", "happy mood", "tense mood", "gloomy mood", "warm mood"]
             
             frame_embedding = clip_model.encode([frame_to_classify], convert_to_numpy=True, show_progress_bar=False)[0]
             style_embeddings = clip_model.encode(style_prompts, convert_to_numpy=True, show_progress_bar=False)
             mood_embeddings = clip_model.encode(mood_prompts, convert_to_numpy=True, show_progress_bar=False)
             
             style_sim = cosine_similarity([frame_embedding], style_embeddings)[0]
             mood_sim = cosine_similarity([frame_embedding], mood_embeddings)[0]
             
             visual_style = style_prompts[np.argmax(style_sim)]
             mood_estimation = mood_prompts[np.argmax(mood_sim)]
             logger.debug(f"Estimated style/mood for {os.path.basename(video_path)}: {visual_style}, {mood_estimation}")
         except Exception as e:
             logger.error(f"Error during CLIP style/mood estimation for {video_path}: {e}")
             visual_style = "[Style/Mood Error]"
             mood_estimation = "[Style/Mood Error]"
             
     # --- Camera Motion (Placeholder) ---
     # Real implementation requires optical flow analysis, which is complex
     camera_motion = "[Not implemented]"
     
     return {
         "visual_style": visual_style,
         "mood_estimation": mood_estimation,
         "camera_motion": camera_motion,
         "estimated_people_count": people_count_str
     }

# --- Main Label Generation Function (Updated) ---
def generate_labels(video_path, technical_metadata):
    """
    Main function to generate REAL AI labels for a video.
    Takes the video path and existing technical metadata.
    Args/Returns are the same as the placeholder version.
    """
    if not technical_metadata:
        logger.error(f"Cannot generate AI labels without technical metadata for {video_path}")
        return None
    
    # Ensure models are loaded (might be called multiple times, load_model handles check)
    initialize_models() # Call this to ensure models are ready in the current process

    logger.info(f"Generating REAL AI labels for: {os.path.basename(video_path)}")
    start_label_time = time.time()

    # --- Call Real Labeling Functions ---
    logger.info("  Running Captioning...")
    caption = generate_video_caption_real(video_path)
    
    logger.info("  Running Object Detection...")
    objects = detect_objects_real(video_path)
    
    logger.info("  Running Action Recognition...")
    actions = recognize_actions_real(video_path)
    
    logger.info("  Running Scene Classification...")
    scene_info = classify_scene_real(video_path)
    
    logger.info("  Running Audio Analysis...")
    audio_info = analyze_audio_real(video_path)
    
    logger.info("  Running Additional Metadata...")
    additional_meta = generate_additional_metadata_real(video_path, objects)

    # --- Combine Results ---
    combined_metadata = technical_metadata.copy()
    combined_metadata.update({
        "ai_caption": caption,
        "ai_objects": objects, # Dict {label: confidence}
        "ai_actions": actions, # List of top actions
        "ai_scene_type": scene_info.get("scene_type", "[Error]"),
        "ai_environment": scene_info.get("environment", "[Error]"),
        "ai_has_speech": audio_info.get("has_speech", False),
        "ai_language": audio_info.get("language"),
        "ai_sound_events": audio_info.get("sound_events", ["[Error]"]),
        "ai_transcription_snippet": audio_info.get("transcription_snippet"),
        "ai_visual_style": additional_meta.get("visual_style", "[Error]"),
        "ai_mood_estimation": additional_meta.get("mood_estimation", "[Error]"),
        "ai_camera_motion": additional_meta.get("camera_motion", "[Not implemented]"),
        "ai_estimated_people_count": additional_meta.get("estimated_people_count", "[Error]"),
        "ai_labeling_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    })
    
    labeling_duration = time.time() - start_label_time
    logger.info(f"Finished generating labels for: {os.path.basename(video_path)} in {labeling_duration:.2f}s")
    return combined_metadata

# --- Main function for testing this module --- 
# (Note: Running this directly can be slow and memory intensive)
if __name__ == "__main__":
    # Example usage: Process metadata files generated by preprocessing.py

    metadata_input_dir = "data/metadata_output"
    labeled_output_dir = "data/labeled_output_real"

    # Initialize models (can take time)
    initialize_models()

    if not os.path.exists(labeled_output_dir):
        os.makedirs(labeled_output_dir)
        logging.info(f"Created directory: {labeled_output_dir}")

    # Find .json files
    try:
         metadata_files = [
             os.path.join(metadata_input_dir, f)
             for f in os.listdir(metadata_input_dir)
             if f.endswith('.json') and not f.startswith('_all')
         ]
    except FileNotFoundError:
         print(f"Error: Metadata input directory not found: '{metadata_input_dir}'")
         print("Run preprocessing.py first.")
         metadata_files = []
         # Optionally create dummy for structure testing
         if not os.path.exists(metadata_input_dir): os.makedirs(metadata_input_dir)
         dummy_meta_path = os.path.join(metadata_input_dir, "dummy_meta.json")
         if not os.path.exists(dummy_meta_path):
             # Need a real filepath even if it doesn't exist for the code path
             dummy_video_path = os.path.abspath("data/input_videos/dummy_video_for_labeling.mp4") 
             dummy_meta_content = {"filename": os.path.basename(dummy_video_path), "filepath": dummy_video_path, "duration_seconds": 1.0, "width": 10, "height": 10}
             with open(dummy_meta_path, 'w') as f: json.dump(dummy_meta_content, f)
             logging.info(f"Created dummy metadata file: {dummy_meta_path}")
             metadata_files.append(dummy_meta_path)
             # Ensure dummy video dir exists too for path resolution
             if not os.path.exists(os.path.dirname(dummy_video_path)): os.makedirs(os.path.dirname(dummy_video_path))
             print("\n--- NOTE: Using dummy metadata. Real labeling will likely fail without actual video files. ---")

    if not metadata_files:
        if os.path.exists(metadata_input_dir):
             print(f"No individual metadata JSON files found in '{metadata_input_dir}'. Run preprocessing.py first.")
        exit()

    all_labeled_data = []
    print(f"Found {len(metadata_files)} metadata files. Generating REAL AI labels (this may take time)...")

    for meta_file in metadata_files:
        try:
            with open(meta_file, 'r') as f:
                technical_metadata = json.load(f)
            
            video_path = technical_metadata.get('filepath')
            if not video_path:
                 logging.warning(f"Skipping {meta_file}: missing 'filepath' key.")
                 continue
                 
            # Check if the actual video file exists for real processing
            if not os.path.exists(video_path):
                 logging.error(f"Video file not found: {video_path}. Cannot generate real labels. Skipping.")
                 continue
            if os.path.getsize(video_path) == 0:
                 logging.warning(f"Video file is empty: {video_path}. Skipping real labeling.")
                 continue

            print(f"Labeling: {os.path.basename(video_path)}")
            labeled_data = generate_labels(video_path, technical_metadata)

            if labeled_data:
                all_labeled_data.append(labeled_data)
                output_filename = os.path.join(labeled_output_dir,
                                               os.path.splitext(os.path.basename(video_path))[0] + "_real_labeled.json")
                try:
                    with open(output_filename, 'w') as f:
                        json.dump(labeled_data, f, indent=4)
                    logging.debug(f"Saved labeled data to {output_filename}")
                except IOError as e:
                     logging.error(f"Failed to save labeled JSON for {video_path}: {e}")
            else:
                 print(f"  -> Failed to generate labels for {os.path.basename(video_path)}")

        except json.JSONDecodeError:
            logging.error(f"Skipping invalid JSON file: {meta_file}")
        except Exception as e:
            # Catch-all for unexpected errors during the loop for one file
            logging.error(f"Unexpected error processing metadata file {meta_file}: {e}", exc_info=True)

    print("\n--- Real Labeling Summary ---")
    print(f"Attempted labeling for {len(metadata_files)} files.")
    print(f"Successfully generated labels for {len(all_labeled_data)} videos.")

    if all_labeled_data:
        all_labeled_path = os.path.join(labeled_output_dir, "_all_real_labeled_data.jsonl")
        try:
             with open(all_labeled_path, 'w') as f:
                 for record in all_labeled_data:
                     # Convert numpy types for JSON serialization if necessary
                     f.write(json.dumps(record, default=lambda x: x.item() if isinstance(x, np.generic) else x) + '\n')
             print(f"Saved combined labeled data to {all_labeled_path}")
        except IOError as e:
             logging.error(f"Failed to save combined labeled data file: {e}")

        print("\nExample Labeled Data (Caption & Action):")
        for i, data in enumerate(all_labeled_data[:3]): # Show first 3
            print(f"- {data['filename']}: Cap='{data.get('ai_caption', 'N/A')}', Actions={data.get('ai_actions', 'N/A')}") 