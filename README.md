# Delay Prediction GNN – README

## 1. Data Filtering and Preparation

This stage covers the complete data filtering and preparation pipeline. Due to the complexity and heterogeneity of GTFS data, this process required deep insight into the data formats and extensive exploratory data analysis to identify and handle edge cases and systematic issues (e.g. *midnight cliff*, incomplete trips, ad hoc services).

---

### Static GTFS Processing
- Retain **only records related to bus routes**
- All other transportation modes are filtered out at this stage

---

### Dynamic GTFS Processing
- From approximately  
  **14 × 24 × 60** `feed_<yyyymmdd>_<hhmmss>.pb` files per period,
  aggregate data into roughly  
  **14 `<yyyymmdd>.csv` files** for efficient downstream processing
- Remove unusable or corrupted records
- Filter out **ad hoc trips** that are not present in the static GTFS dataset
- Split days into **training** and **testing** sets  
  - The split configuration is defined in:  
    `shared/data/records.json`

---

### Delay Calculation
- Based on the **filtered static and dynamic GTFS data**
- Compute **end-of-trip delays** for all tracked trips (when available)
- Stored attributes:
  - `route_id`
  - `trip_id`
  - `delay`
  - `t_start`

---

### Graph Construction

#### Static Graph Features (from Static GTFS)
- Outlier filtering
- **Edges**:
  - `distance`
  - `average_travel_time`
- **Nodes**:
  - `longitude`
  - `latitude`

#### Temporal Binning
- Each day is split into **30-minute time bins**

#### Dynamic Graph Features (from Dynamic GTFS)
- Outlier filtering per time bin
- **Node-level features** for each stop and time bin:
  - `count`
  - `mean`
  - `std`
  - `max`
  - `min`  
  (statistics of trips passing through the stop in the given interval)

---

### Resulting Data Artifacts
- `delays/<yyyymmdd>.csv`  
  Contains end-of-trip delays and trip start times for all tracked trips on the given day
- `graphs/<yyyymmdd>/graph_<bin_code>.pt`  
  PyTorch graph object containing static and dynamic features for the specified day and time bin

## 2. Baseline Model

As a baseline, a simple data-driven approach was implemented using only the aggregated delay information from the `delays/<yyyymmdd>.csv` files, without any graph structure or neural networks. The goal was to establish a lower-bound performance reference.

---

### Data Selection
- For each **day type** (Monday, Tuesday, …):
  - One `delays/<yyyymmdd>.csv` file was selected for training
  - One `delays/<yyyymmdd>.csv` file was selected for testing
- This setup ensures a fair comparison across different weekday patterns

---

### Baseline Variants

#### Route-Based Model
- Prediction strategy:
  - Compute the **mean delay per route**, aggregated over the training day
- Evaluation result:
  - **MAE ≈ 88.87 s**

---

#### Route-Period Model
- Prediction strategy:
  - Compute the **mean delay per route and time period**, aggregated over the training day
- Time periods:
  - `[0, 6, 9, 15, 18, 24]` hours
- Evaluation result:
  - **MAE ≈ 86.21 s**

---

### Remarks
- These baselines rely solely on historical averaging
- No spatial, temporal continuity, or interaction effects are modeled
- The results provide a strong reference point for evaluating the added value of graph-based and learning-based approaches

## 3. Feature Scaling Strategy

To ensure stable and successful model training, feature scaling was applied consistently across all training data. The scaling process was designed to avoid data leakage, preserve temporal structure, and handle known GTFS-specific issues such as the *midnight cliff*.

---

### Fitting Procedure
- All scaling parameters were fitted **exclusively on training days** (`<yyyymmdd>`)
- Inputs used for fitting:
  - `delays/<yyyymmdd>.csv` (end-of-trip delays and trip start times)
  - `delays/<yyyymmdd>/graph_<bin_code>.pt` (graph-structured data)
- A **StandardScaler** was fitted separately for:
  - Node features
  - Edge features
  - Target variable (`delay`)
- Fitted scalers were **saved and reused** during validation and inference to guarantee training–inference parity

---

### Temporal Features
In addition to graph features, temporal context was incorporated using **day type** and **time-of-day** information. Two alternative encoding strategies were evaluated:

#### Min–Max Encoding
- Day of week mapped to `[-1, 1]`  
  - Monday → `-1`, Sunday → `1`
- Time of day mapped to `[-1, 1]`  
  - `0h → -1`, `24h → 1`
- Resulting in **2 temporal features**

#### Periodic Encoding (Midnight-Cliff Safe)
- Sine and cosine encoding to preserve periodic continuity
- Features:
  - `sin`, `cos` with **weekly period** (day of week)
  - `sin`, `cos` with **daily period** (time of day)
- Resulting in **4 temporal features**
- This representation explicitly avoids discontinuities at day/week boundaries

---

## 4. Model development

The first learning-based approach leverages both **trip-level delay data** and **time-dependent graph representations**. The core idea is to condition the prediction on the trip identity while extracting spatiotemporal context from the corresponding graph snapshot.

---

### Inputs
- **Trip identifiers**: `day`, `trip_id`
- From `trip_id`, the following are implicitly available:
  - Trip start time
  - Ordered stop sequence

---

### Graph Selection
- The trip start time and day are used to select the corresponding
  **30-minute graph snapshot** of that day
- This graph encodes the dynamic traffic state for the relevant time window

---

### Graph Feature Extraction
- The selected graph is processed using a stack of **NNConv layers**
- These layers aggregate information from neighboring nodes, conditioned on edge features

---

### Trip-Level Pooling
- From the resulting node embeddings:
  - **Average pooling** is performed over the nodes belonging to the trip’s stop sequence
  - The stop sequence defines a subgraph-specific node set
- This yields a fixed-size **trip-level feature vector**

---

### Prediction Head
- The pooled graph features are concatenated with:
  - Encoded day information
  - Encoded trip start time
- The resulting feature vector is passed through an **MLP**
- Output:
  - Predicted **end-of-trip delay**

---

### Motivation
- Captures **spatial dependencies** via graph convolutions
- Incorporates **temporal context** through time-dependent graph snapshots
- Aligns naturally with GTFS semantics (trip → stops → route)

This strategy serves as the baseline graph-neural formulation upon which more advanced temporal and sequence-aware models can be built.


## 5. Data Handling

This section describes the current data-loading and batching strategy used during training, along with identified limitations and planned improvements.

---

### Current Training-Time Data Flow
- During each epoch, training iterates **randomly over the recorded training days** (`<yyyymmdd>`)
- For a selected day:
  - Load `delays/<yyyymmdd>.csv` into memory
  - Load all available graphs from  
    `graphs/<yyyymmdd>/graph_<bin_code>.pt` into a cache

---

### Trip Filtering
- Trips are filtered out if:
  - Their scheduled start time does not correspond to any available  
    `graph_<bin_code>.pt` file for that day
- This ensures that each remaining trip has a valid graph representation

---

### Batching Strategy
- After filtering, trips are grouped into batches
- Graph batching is handled via  
  `torch_geometric.loader.DataLoader`
- This merges multiple graphs into a single batched graph
- Special care is required to:
  - Pool node embeddings **per trip**
  - Correctly handle index offsets introduced by graph batching

---

### Limitations
- Each trip is associated with a **single 30-minute graph snapshot**
- Traffic conditions evolving during the trip are not explicitly modeled

---

### Planned Improvement
- Load the original `<yyyymmdd>.csv` used for full data generation
- For each trip:
  - Query or construct a **composite graph** that integrates traffic information over the entire trip duration
  - Cover the interval between the scheduled **start and end times**
- This would allow the model to capture **within-trip temporal dynamics** rather than relying on a single time bin

## 6. Evaluation and Inference

This section describes the procedures for evaluating the model on reserved data and the differences with real-world inference scenarios.

---

### Evaluation Procedure
- Reserved evaluation data is stored separately from training data
- Iteration over evaluation trips follows the **same procedure as training**:
  - Load the corresponding `delays/<yyyymmdd>.csv`
  - Load relevant graph snapshots from `graphs/<yyyymmdd>/graph_<bin_code>.pt`
  - Filter trips without matching graphs
- Predictions are made using the **actual traffic state graphs** corresponding to the trip start times

---

### Inference Considerations
- **Key difference from evaluation**:
  - In real-world inference, future traffic data is **not available**
- Strategy for inference:
  - Use a past graph snapshot that matches the **same day of week** and **time interval**
  - This serves as an approximation of expected traffic conditions based on historical patterns
- This approach ensures the model can generate realistic delay predictions even without live traffic feeds



## File Structure and Functions

The repository is organized to clearly separate data processing, model development, training, and experimentation. The structure is designed to support reproducibility, scalability, and efficient experimentation.

---

### `src/` — Active Scripts (Entry Points)
Contains executable scripts that define the main workflow of the project.

- `data-query.py`  
  Fetches raw real-time data from the BKK API.

- `data-filter.py`  
  Filters and organizes the raw dataset (~3.5 GB), producing a compact and structured dataset (~230 MB).

- `preprocess.py`  
  Fits preprocessing scalers on the training data and saves them for consistency.

- `train.py`  
  Implements the main training loop, including data loading, model optimization, and checkpointing.

- `evaluation.py`  
  Handles model evaluation on the test set and computes performance metrics.

- `utils/`  
  Collection of helper functions and shared utilities used across scripts.

---

