## Part 1: Enumerated Outline of Actively Callable Code Files

Below is a structured outline of each Python code file in this project, describing classes, functions, methods, and their essential signatures with concise technical commentary. This includes references to pydantic models, data structures, and external dependencies. The purpose is to give you a bird’s-eye view of the codebase and how its pieces fit together.

---

### 1. **`config.py`**

**Purpose**  
- Loads and validates the YAML configuration file (`config.yaml`).  
- Defines Pydantic models for configuration sections to ensure type safety.

**Key Classes**  
1. **`ExecutionConfig(BaseModel)`**  
   - **Fields**  
     - `max_calls_per_prompt: int`  
     - `batch_size: int`  
     - `nshot_ct: int`  
   - **Description**: Holds execution-related parameters (e.g., how many times a prompt can be called, number of samples in a batch, etc.).

2. **`FlagsConfig(BaseModel)`**  
   - **Fields**  
     - `max_samples: int`  
     - `FLAG_PROMPT_PREFIX: bool`  
     - `FLAG_PROMPT_SUFFIX: bool`  
     - `prompt_prefix: str`  
     - `prompt_suffix: str`  
   - **Description**: Holds various flags and prefix/suffix configurations that can be toggled for prompt engineering.

3. **`Config(BaseModel)`**  
   - **Fields**  
     - `model_parameters: Dict[str, Any]`  
     - `execution: ExecutionConfig`  
     - `flags: FlagsConfig`  
     - `timeout: Dict[str, Any]`  
     - `logging: Dict[str, Any]`  
     - `output: Dict[str, Any]`  
     - `data: Dict[str, Any]`  
     - `prompts: Dict[str, str]`  
     - `model_ensemble: Dict[str, Dict[str, Any]]`  
   - **Properties**  
     - `max_samples -> int`  
     - `max_calls_per_prompt -> int`  
     - `batch_size -> int`  
     - `nshot_ct -> int`  
   - **Description**: Aggregates all sub-sections of configuration into a single, validated object. Exposes convenience properties to retrieve common parameters.

**Key Functions**  
1. **`load_config(config_file: str = "config.yaml") -> Config`**  
   - **Docstring**:  
     Loads configuration from the specified YAML file and returns a validated `Config` object.  
   - **Exceptions**:  
     - `ValidationError` if YAML schema doesn't match the Pydantic models.  

---

### 2. **`setup.py`**

**Purpose**  
- Standard Python packaging script using setuptools.

**Key Callable**  
1. **`setup(...)`**  
   - **Description**:  
     Configures the distribution/installation parameters for this package (i.e., metadata, required dependencies, and classifiers).  

**Notable Dependencies**  
- `setuptools`, `find_packages`  

_(Typically not directly invoked in runtime code, but used when installing or distributing the package.)_

---

### 3. **`models.py`**

**Purpose**  
- Defines the data models for prompt types, risk factors, decisions, etc. using Pydantic for validation.

**Key Enums**  
1. **`PromptType(str, Enum)`**  
   - **Values**  
     - `SYSTEM1`  
     - `COT`  
     - `COT_NSHOT`  
   - **Description**: Defines valid prompt types used by the system.

2. **`RiskWeight(str, Enum)`**  
   - **Values**  
     - `HIGH`  
     - `MEDIUM`  
     - `LOW`  
   - **Description**: Levels for risk factor weighting.

**Key Classes**  
1. **`RiskFactor(BaseModel)`**  
   - **Fields**  
     - `factor: str`  
     - `weight: RiskWeight`  
     - `reasoning: str`  
   - **Description**: Represents an individual risk factor with a textual explanation.

2. **`Decision(BaseModel)`**  
   - **Fields**  
     - `prediction: str` (`pattern="^(YES|NO)$"`)  
     - `confidence: int` (range: `0–100`)  
   - **Description**: Minimal decision object containing the predicted label and confidence score.

3. **`DecisionWithRiskFactors(Decision)`**  
   - **Additional Field**  
     - `risk_factors: List[RiskFactor]`  
   - **Description**: Extended version of `Decision` that also includes analysis of risk factors.

---

### 4. **`prompt_manager.py`**

**Purpose**  
- Dynamically generates prompts for the language model by inserting context from the data and handling different prompt types.

**Key Classes**  
1. **`PromptManager`**  
   - **Constructor**  
     - `__init__(self, config: Config, data_manager: DataManager) -> None`  
       - Stores references to the global `Config` object and a `DataManager`.  
   - **Private Methods**  
     - `def _generate_nshot_examples(self) -> str`  
       - Generates N-shot examples from the validation set by sampling the data.  
       - Caches the generated examples to avoid repeated computation.  
   - **Public Methods**  
     1. **`get_prompt(self, prompt_type: PromptType, row_id: int) -> str`**  
        - **Docstring**:  
          Retrieves a prompt template, inserts relevant risk factors, and optionally appends n-shot examples if the prompt type is `COT_NSHOT`.  
        - **Dependencies**:  
          - Relies on `data_manager` to fetch risk factors and sample data for n-shot.  
        - **Exceptions**:  
          - `ValueError` if the prompt type is invalid.  
          - Propagates exceptions from data fetching.  

---

### 5. **`metrics.py`**

**Purpose**  
- Defines data structures for tracking and recording execution metrics, such as timeouts, error messages, confidence scores, etc.

**Key Classes**  
1. **`TimeoutMetrics`** (`@dataclass`)  
   - **Fields**  
     - `occurred: bool`  
     - `retry_count: int`  
     - `total_timeout_duration: float`  
     - `final_timeout_duration: Optional[float]`  

2. **`PromptMetrics`** (`@dataclass`)  
   - **Fields**  
     - `attempt_number: int`  
     - `execution_time_seconds: float`  
     - `successful: bool`  
     - `timeout_metrics: TimeoutMetrics`  
     - `error_message: Optional[str]`  
     - `prediction: Optional[str]`  
     - `confidence: Optional[float]`  
     - `meta_data: Optional[Dict[str, Any]]`  
   - **Methods**  
     - `def __post_init__(self) -> None`:  
       Ensures type conversions for certain fields (e.g., `execution_time_seconds` is always float).

3. **`TimeoutStats`** (`@dataclass`)  
   - **Fields**  
     - `total_timeouts: int`  
     - `avg_timeout_duration: float`  
     - `max_timeout_duration: float`  
     - `total_timeout_duration: float`  

---

### 6. **`performance.py`**

**Purpose**  
- Records performance statistics across multiple prompt attempts or sessions and produces consolidated metrics and reports (JSON, text, etc.).

**Key Classes**  
1. **`PerformanceStats`** (`@dataclass`)  
   - **Fields** (high-level)  
     - `prompt_type, model_name, start_time, end_time, total_attempts, successful_attempts, …`  
     - `timeout_stats: TimeoutStats`  
     - `prediction_accuracy: float`  
     - `prediction_distribution: Dict[str, int]`  
     - `confusion_matrix: Dict[str, int]`  
     - `auc_roc: float`  
   - **Description**: Stores computed statistics for an evaluation session, including confusion matrix, AUC-ROC, timings, etc.

2. **`DecisionTracker`**  
   - **Purpose**: Aggregates predictions (YES/NO), compares them with actual outcomes, and computes accuracy.  
   - **Key Methods**  
     1. **`record_prediction(self, prediction: str, actual: str, confidence: Optional[float]) -> None`**  
        - Maintains count of correct/incorrect predictions and stores confidence for later metrics.  
     2. **`get_accuracy(self) -> float`**  
        - Returns overall accuracy percentage.  
     3. **`get_stats(self) -> Dict[str, Any]`**  
        - Returns distribution of predictions, accuracy, and raw vectors for actual vs. predicted labels.

3. **`PerformanceTracker`**  
   - **Purpose**: Encapsulates a single session or run of prompt attempts; collects `PromptMetrics`, aggregates them into final `PerformanceStats`.  
   - **Key Methods**  
     1. **`record_attempt(self, metrics: PromptMetrics) -> None`**  
        - Logs each attempt’s success/failure, timing, and calls `DecisionTracker` if the prediction is valid.  
     2. **`save_metrics(self, execution_time: float) -> None`**  
        - Computes final stats via `_generate_stats()` and writes them to disk in JSON and text form.  

**Utility Function**  
1. **`save_aggregate_stats(session_results: List[PerformanceStats], total_duration: float) -> None`**  
   - Gathers multiple session results into a single aggregate JSON and text report (useful if multiple prompt types/models are tested in one run).

---

### 7. **`data_manager.py`**

**Purpose**  
- Handles reading from the CSV input file, normalizing target labels, stratified splitting into train/validation/test sets, and providing data retrieval (risk factors, actual values, batch samples).

**Key Class**  
1. **`DataManager`**  
   - **Constructor**  
     - `__init__(self, config: Config) -> None`  
       - Initializes placeholders for dataframes and sets path from `config`.  
   - **Private Methods**  
     1. **`_normalize_target_values(self, df: pd.DataFrame) -> pd.DataFrame`**  
        - Converts target column into consistent `YES/NO` format.  
        - Raises `ValueError` if encountering unrecognized labels.  
   - **Public Methods**  
     1. **`validate_data(self, df: pd.DataFrame) -> None`**  
        - Checks for required columns, missing values, and valid target distribution.  
     2. **`load_and_prepare_data(self) -> Tuple[int, int]`**  
        - Reads CSV, normalizes targets, splits data into train/validate/test, ensuring stratification.  
        - Returns the sizes of the train and test sets.  
     3. **`get_batch(self, batch_size: int, dataset: str = 'train') -> List[Dict[str, Any]]`**  
        - Samples a random mini-batch from `train`, `test`, or `validate`.  
        - Returns list of dictionaries containing `id`, `risk_factors`, `target`.  
     4. **`get_risk_factors(self, row_id: int) -> str`**  
        - Retrieves the risk factor text for a given row ID from the training set.  
     5. **`get_actual_value(self, row_id: int) -> str`**  
        - Retrieves the ground-truth label (`YES` or `NO`) for a given row ID from the training set.  
     6. **`get_dataset_info(self) -> Dict[str, Any]`**  
        - Returns details about dataset sizes and simple feature statistics.

---

### 8. **`decision.py`**

**Purpose**  
- Orchestrates calls to the LLM (via `ollama`), processes the response to a structured JSON decision, and saves decision metadata to disk.

**Key Classes**  
1. **`MetaData(BaseModel)`**  
   - **Fields**  
     - `model: Optional[str]`  
     - `created_at: Optional[str]`  
     - `done_reason: Optional[str]`  
     - `done: Optional[bool]`  
     - `total_duration: Optional[float]`  
     - `load_duration: Optional[float]`  
     - `prompt_eval_count: Optional[int]`  
     - `prompt_eval_duration: Optional[float]`  
     - `eval_count: Optional[int]`  
     - `eval_duration: Optional[float]`  

**Key Functions**  
1. **`generate_output_path(...) -> Path`**  
   - Constructs a file path for saving model decisions based on model name, prompt type, row ID, and a timestamp.

2. **`process_model_response(response_text: str) -> Dict`**  
   - Attempts multiple parsing strategies (direct JSON, regex extraction, fallback parsing) to decode the LLM response into `{'prediction': 'YES'|'NO', 'confidence': 0..100, ...}`.  
   - **Raises**: `ValueError` if it cannot parse a valid JSON response.

3. **`get_decision(...) -> Tuple[Optional[Decision], Optional[MetaData], str, Dict[str, Any]]`** (asynchronously)  
   - **Docstring**:  
     Sends a prompt to the LLM and returns a structured `Decision` plus any meta-data gleaned from the model.  
   - **Dependencies**:  
     - `ollama.chat` for LLM calls.  
   - **Exceptions**:  
     - `ValidationError` if the parsed response is invalid for the `Decision` schema.

4. **`get_decision_with_timeout(...) -> Tuple[Optional[Decision], Optional[MetaData], str, Dict[str, Any], TimeoutMetrics]`** (asynchronously)  
   - Wraps `get_decision` in a `asyncio.wait_for()` to handle timeouts, updating `TimeoutMetrics` if the call does not return in time.

5. **`save_decision(...) -> bool`**  
   - Serializes the `Decision` and associated metadata to JSON on disk, using the path from `generate_output_path`.  
   - Returns `True` if save is successful, otherwise `False`.

---

### 9. **`main.py`**

**Purpose**  
- Intended entry point for the application (no detailed code given). Typically orchestrates the steps below:
  1. Load configuration  
  2. Initialize `DataManager` and `PromptManager`  
  3. Run LLM decisions (e.g., calling `get_decision_with_timeout`)  
  4. Track performance metrics  
  5. Save or aggregate results  

(No direct code is shown here, so we assume it imports and calls the modules described above.)

---

## Part 2: Professional Technical Document

Below is an overview of the system’s architecture, control flow, design choices, and recommended future improvements. This document is intended for developers or stakeholders who need to understand the higher-level workings of the code and may be called upon to maintain or extend it.

---

### **1. Overall Architecture**

1. **Configuration-Driven**  
   - All global parameters are stored in `config.yaml` and validated by `config.py`. This design allows advanced users to modify model and data settings without altering the code.

2. **Layered Components**  
   - **Data Management** (`data_manager.py`): Reading CSV, normalizing labels, splitting data.  
   - **Modeling / Decision** (`decision.py`, `models.py`): Interfacing with the LLM to generate predictions and structured output.  
   - **Prompt Engineering** (`prompt_manager.py`): Assembling prompt templates and injecting dynamic content (risk factors, n-shot examples, etc.).  
   - **Metrics & Performance** (`metrics.py`, `performance.py`): Tracking timeouts, success rates, confusion matrices, and storing results.

3. **Asynchronous LLM Calls**  
   - The system uses `asyncio` to handle the possibility of timeouts for LLM responses. This fosters resilience in environments with high latency or unstable service connections.

4. **Data Validation**  
   - Pydantic is used heavily to ensure type correctness and data validation at multiple stages (configuration, model responses, etc.), minimizing run-time errors.

---

### **2. Control Flow**

1. **Initialization**  
   - **Step 1**: Load the YAML config via `load_config()` which returns a `Config` object.  
   - **Step 2**: Instantiate `DataManager` with the `Config`, then call `load_and_prepare_data()` to create train/validate/test splits.

2. **Prompt Preparation**  
   - **Step 3**: Instantiate `PromptManager`, which loads prompt templates from `config.prompts`.  
   - **Step 4**: For each sample to be tested, call `PromptManager.get_prompt(...)` with a `PromptType` and a `row_id`, receiving a text prompt with risk factors embedded.

3. **Decision & Model Interaction**  
   - **Step 5**: Use `decision.py` to call the LLM with `get_decision_with_timeout(...)`, passing the prompt, model name, etc.  
   - **Step 6**: The LLM’s raw text response is parsed by `process_model_response()`, which attempts to produce a well-formed JSON.  
   - **Step 7**: A `Decision` object is constructed and validated. If successful, it is saved to disk along with additional `MetaData`.

4. **Tracking & Reporting Performance**  
   - **Step 8**: `PerformanceTracker` records each attempt via `record_attempt()`.  
   - **Step 9**: Once the session completes, `PerformanceTracker.save_metrics()` generates a detailed JSON and text-based report. Multiple sessions may be aggregated by calling `save_aggregate_stats()`.  

---

### **3. Design Choices and Tradeoffs**

1. **Use of Pydantic**  
   - **Pro**: Strong validation and clarity of data shapes; helps maintain correctness in dynamic AI workflows.  
   - **Con**: Some overhead in performance and complexity for developers unfamiliar with Pydantic.

2. **Async Code for LLM Calls**  
   - **Pro**: Allows timeouts and concurrency to be handled gracefully.  
   - **Con**: Introduces complexity in debugging and structure (event loop usage, partial concurrency).

3. **N-Shot Example Generation**  
   - Randomly sampling examples from the validation set can be effective for instructing the model.  
   - For deterministic reproducibility, the random seed is set from `config.data["random_seed"]`.

4. **Stratified Splits**  
   - Ensures balanced distribution of classes (`YES` vs. `NO`) in training, validation, and test sets.  
   - Potential tradeoff if the dataset is very small—some classes could still be underrepresented.

5. **Timeout and Retries**  
   - The code collects `TimeoutMetrics` to handle LLM timeouts gracefully.  
   - If timeouts happen frequently, the developer must fine-tune the concurrency strategy or handle partial results.

6. **Performance Tracking**  
   - Real-time metrics capturing (e.g., AUC-ROC, confusion matrix, accuracy).  
   - The approach is flexible but might create large amounts of logging artifacts if scaled to big datasets.

---

### **4. Possible Future Improvements**

1. **Enhanced Prompt Templating**  
   - Currently, string replacement is used. Introducing a dedicated templating system (like Jinja2) might improve readability and maintainability for complex prompts.

2. **Additional Fairness Scenarios**  
   - The code contains commented-out “scenario_fairness_…” text blocks. Future expansions might integrate these fairness constraints into the model’s prompt logic or post-processing.

3. **Caching Mechanism**  
   - For repeated runs, a caching solution (e.g., JSON caching of LLM responses) could greatly reduce inference cost during development and debugging cycles.

4. **Error Handling and Retries**  
   - The system already collects metrics. Implementing automatic re-tries for certain error types (beyond just timeouts) could increase robustness.

5. **Extension to Multiple Data Sources**  
   - If other data sources (databases, JSON files) or multiple CSVs are used, the DataManager can be abstracted further for more flexible data ingestion.

6. **Web Interface / Monitoring**  
   - The text and JSON logs can be served in a small web dashboard, giving real-time insights into model performance, queue statuses, and timeouts.

---

### **5. Conclusion**

This codebase provides a solid framework for large language model (LLM) decision-making, risk assessment analysis, and thorough performance tracking. By leveraging carefully designed Pydantic models, prompt management routines, and detailed logging, it offers both reliability and transparency. The modular structure allows new team members to quickly understand where each piece fits: from data ingestion to final decision saving and metric reporting.

Going forward, expansions such as more sophisticated prompt templating, additional fairness policies, or a user-facing dashboard could raise the utility and maintainability of this system. Overall, the project demonstrates robust engineering practices while remaining open to further enhancements by future developers.