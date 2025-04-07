# Video Data Processing Pipeline

This project implements a scalable pipeline for crawling, deduplicating, processing, and labeling video data, as described in `data-collection.md`.

## Local Test Pipeline Flow

```mermaid
graph TD
    A[Start Pipeline] --> B{Scan Input Dir};
    B --> C[Video Files List];
    C --> D{Calculate Hashes (Parallel)};
    D --> E[Hash Results];
    E --> F{Identify Unique/Duplicates};
    F -- Unique Videos --> G{Extract Metadata (Parallel)};
    F -- Duplicates --> H[Store Duplicate Map];
    G --> I[Metadata Results];
    I --> J{Generate Labels (Parallel)};
    J --> K[Labeled Data Results];
    K --> L{Save Results};
    L --> M[JSONL Output];
    L --> N[CSV Output];
    H --> L;  // Duplicate map is also saved at the end
    L --> Z[End Pipeline];

    subgraph Ingestion [src/ingestion.py]
    B
    end

    subgraph Deduplication [src/deduplication.py + pipeline.py]
    D
    F
    H
    end

    subgraph Preprocessing [src/preprocessing.py]
    G
    end

    subgraph Labeling [src/labeling.py]
    J
    end
    
    subgraph Output [pipeline.py]
    L
    M
    N
    end
```

## Local Test Pipeline

A simplified version of the pipeline can be run locally to test the core logic using sample video files. This version uses:
*   Local file system ingestion (`src/ingestion.py`)
*   Perceptual hashing for basic deduplication (`src/deduplication.py`)
*   `ffmpeg` for technical metadata extraction (`src/preprocessing.py`)
*   Placeholder functions for AI labeling (`src/labeling.py`)
*   Ray for local orchestration (`src/pipeline.py`)

### Setup

1.  **Clone the repository (if you haven't already).**
2.  **Create and activate a Python environment:**
    ```bash
    # Using conda (recommended)
    conda create -n video_pipeline python=3.10 -y
    conda activate video_pipeline 
    # Or using venv
    # python3 -m venv venv
    # source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **(Important)** Install `ffmpeg`. This library is required by `ffmpeg-python` and `opencv-python`. Installation methods vary by OS:
    *   **macOS (using Homebrew):** `brew install ffmpeg`
    *   **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
    *   **Windows:** Download from the official FFmpeg website and add it to your system's PATH.
5.  **(Optional but Recommended)** Place some sample video files (e.g., `.mp4`, `.mov`, `.avi`) into the `data/input_videos` directory. Include some identical videos and some different ones to test the deduplication step properly. The scripts will create dummy files if the directory is empty, but processing won't be meaningful.

### Running the Local Pipeline

Ensure your Python environment (e.g., `video_pipeline`) is activated.

**Using Perceptual Hashing (Default):**

```bash
python src/pipeline.py --input-dir data/input_videos --output-dir data/pipeline_output_phash
```

**Using Embeddings:**

*Requires `torch`, `sentence-transformers`, `scikit-learn` (`pip install -r requirements.txt`). The first run will download the CLIP model (~500MB).*

```bash
python src/pipeline.py --dedup-method embedding --embedding-threshold 0.90 --input-dir data/input_videos --output-dir data/pipeline_output_embedding
```

**Using Real AI Models for Labeling:**

To use the actual AI models implemented in `src/labeling.py` instead of placeholders, add the `--use-real-labels` flag. 

**Warning:** This is computationally intensive, requires significant RAM and disk space for models, and strongly benefits from a CUDA-enabled GPU. Processing will be much slower than with placeholders.

```bash
# Example using real labels and embedding deduplication (requires GPU for reasonable speed)
python src/pipeline.py --dedup-method embedding --use-real-labels --input-dir data/input_videos --output-dir data/pipeline_output_real_labels
```

**Optional Arguments:**

*   `--input-dir`: Specify input directory.
*   `--output-dir`: Specify output directory.
*   `--dedup-method`: Choose `phash` or `embedding`.
*   `--phash-size`: Hash size for `phash` method (default: 8).
*   `--phash-threshold`: Max distance for `phash` method (default: 5).
*   `--embedding-threshold`: Min similarity for `embedding` method (default: 0.95).
*   `--num-frames-dedup`: Number of frames to sample for deduplication (default: 5).
*   `--num-cpus`: Limit Ray CPU usage (default: all available).
*   `--log-level`: Set logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

The pipeline will:
1.  Scan `data/input_videos` for videos.
2.  Calculate hashes and identify unique/duplicate videos.
3.  Extract technical metadata for unique videos.
4.  Generate placeholder AI labels.
5.  Save the results (labeled metadata as JSONL and CSV, duplicate map as JSON) to the `data/pipeline_output` directory.

Check the console output for progress and the `data/pipeline_output` directory for the results.

## Full-Scale Pipeline (TODO)

*   Implement distributed web crawling/downloading.
*   Implement robust deduplication (content fingerprints, embeddings).
*   Integrate actual AI models for labeling (requires GPU resources).
*   Set up distributed orchestration (Ray cluster, Kubernetes).
*   Implement scalable storage (Object storage, Metadata DB, Vector Index).
*   Add monitoring and error handling.

## Documentation (TODO)

*   Add Mermaid diagrams for visualization.
*   Document cloud deployment procedures.
*   Detail the configuration options.

## Cloud Deployment Strategy (Outline - TODO)

Deploying this pipeline at scale (100M+ videos, 1000 H100 GPUs, 10k CPUs) requires a cloud-native architecture. Here is a potential strategy based on the design document:

1.  **Compute Orchestration:**
    *   Use **Kubernetes (K8s)** (e.g., EKS on AWS, GKE on GCP, AKS on Azure) to manage containerized applications across the CPU and GPU nodes.
    *   Deploy **Ray on Kubernetes** to handle the distributed compute scheduling for the pipeline stages, enabling efficient resource utilization and auto-scaling of worker pools based on load.

2.  **Pipeline Stages as K8s Deployments/Ray Actors:**
    *   **Ingestion:** Deploy Scrapy/Nutch crawlers as K8s jobs or Ray actors running on CPU nodes. Use a distributed message queue (e.g., Kafka, AWS SQS, Google Pub/Sub) to pass video URLs/tasks to download workers.
    *   **Downloading:** Scalable downloader workers (Ray actors or K8s pods) on CPU nodes, reading from the queue and writing raw videos to object storage.
    *   **Deduplication (Embedding-based):** GPU-accelerated Ray actors for computing video/audio embeddings. Use a vector database (like Weaviate, Milvus, or Pinecone, possibly deployed within K8s) or Faiss index for similarity search.
    *   **Preprocessing:** CPU-intensive tasks (decoding, frame extraction using FFmpeg/OpenCV) run as Ray actors on CPU nodes. GPU-accelerated decoding (NVDEC) can be used where applicable.
    *   **AI Labeling:** Deploy various AI models (packaged in containers) as GPU-accelerated Ray actors. Assign different models to different GPU node pools if needed. Leverage multi-GPU inference and batching.
    *   **Storage/Indexing:** Workers (Ray actors or K8s jobs) to write metadata to the chosen databases (e.g., PostgreSQL/Cassandra for structured data, Elasticsearch/OpenSearch for text search, Vector DB for embeddings).

3.  **Storage:**
    *   **Raw Videos:** Cloud Object Storage (S3, GCS, Azure Blob Storage) for virtually unlimited, durable storage.
    *   **Structured Metadata:** Managed database service (e.g., RDS, Cloud SQL, Cosmos DB) or self-hosted DB (e.g., Cassandra) running on K8s.
    *   **Text Search Index:** Managed Elasticsearch/OpenSearch service or self-hosted cluster on K8s.
    *   **Vector Embeddings:** Managed Vector Database service or self-hosted (Weaviate/Milvus on K8s).
    *   **Intermediate Data:** Ray's distributed object store for in-memory data transfer between pipeline stages, minimizing disk I/O.

4.  **Workflow Management & Monitoring:**
    *   **High-level orchestration:** Apache Airflow (perhaps running on K8s) could trigger and monitor the overall Ray pipeline DAGs, handle scheduling, and manage retries for higher-level tasks (e.g., ingesting from a specific source).
    *   **Monitoring:** Prometheus and Grafana (deployed via Helm charts on K8s) for metrics collection (Ray metrics, K8s metrics, GPU utilization via DCGM exporter).
    *   **Logging:** Centralized logging solution (e.g., EFK stack - Elasticsearch, Fluentd, Kibana or Loki/Promtail/Grafana) to aggregate logs from all pods/actors.

5.  **Networking:** Ensure high-throughput networking between nodes, especially between CPU nodes preparing data and GPU nodes performing inference, and between workers and storage services.

6.  **Configuration Management:** Use tools like Helm for packaging K8s applications and managing configurations (e.g., model paths, database endpoints, resource requests) via ConfigMaps and Secrets. 

## Running on a Ray Cluster (`pipeline_cluster.py`)

The `src/pipeline_cluster.py` script is designed to run on a pre-existing Ray cluster, leveraging distributed storage and GPU actors for labeling.

### Prerequisites

1.  **Ray Cluster:** A running Ray cluster with appropriate resources (CPU nodes, GPU nodes - ideally matching the scale described).
2.  **Shared Storage:** Input videos accessible via a URI prefix (e.g., `s3://your-bucket/videos/`, `gs://...`, `hdfs://...`, or a shared NFS mount `/mnt/...`) readable by all Ray workers.
3.  **Output Storage:** A writable storage URI prefix for results.
4.  **Dependencies:** All dependencies from `requirements.txt` installed in the Python environment used by the Ray cluster workers.
5.  **Vector Database (for Deduplication):** An external, scalable vector database (Milvus, Weaviate, Pinecone) must be set up. The current script only calculates embeddings; the comparison and unique identification step needs to be implemented against such a database.

### Configuration & Execution

Connect to the Ray cluster head node or a machine that can submit jobs to the cluster.

```bash
# Activate the correct Python environment with dependencies installed
# conda activate video_pipeline # or your cluster environment

# Example command
python src/pipeline_cluster.py \
    --ray-address auto \
    --input-path "s3://your-video-bucket/input-prefix/" \
    --output-path "s3://your-results-bucket/output-prefix/" \
    --dedup-method embedding \
    --num-frames-dedup 10 \
    --num-labeling-workers 950 \ # Adjust based on available GPUs (e.g., slightly less than 1000)
    --log-level INFO 
```

**Key Arguments for `pipeline_cluster.py`:**

*   `--ray-address`: Address of the Ray head node (`auto` often works).
*   `--input-path`: **Required.** URI prefix for input videos (e.g., `s3://...`).
*   `--output-path`: **Required.** URI prefix for output results (e.g., `s3://...`).
*   `--dedup-method`: `embedding` (recommended) or `phash`. Note: Comparison logic is a placeholder.
*   `--num-frames-dedup`: Frames to sample for signature calculation.
*   `--num-labeling-workers`: Number of GPU actors to create for parallel labeling (should match available GPUs).
*   `--log-level`: Logging verbosity.

### Important Considerations

*   **Deduplication Placeholder:** The script calculates signatures but **does not perform the large-scale comparison**. You need to integrate the signature writing and querying steps with a dedicated vector database.
*   **Data Access:** Ray workers need appropriate permissions and configurations (e.g., AWS credentials, GCS keys) to access the specified input/output storage paths.
*   **Error Handling:** Robust error handling, retries, and progress tracking are crucial for long-running, large-scale jobs and would need further enhancement.
*   **Ray Datasets:** For optimal performance and memory efficiency at extreme scale, consider refactoring the pipeline to use `ray.data` for streaming data directly from storage through transformations. 