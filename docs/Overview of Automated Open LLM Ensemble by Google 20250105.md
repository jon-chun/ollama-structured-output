Okay, let's break down this Python project step-by-step.

### 1. Actively Callable Code Files and Technical Outline

Here's a breakdown of the actively callable Python files, their classes, functions, and dependencies:

**I. `config.py`**

*   **Summary:** Handles configuration loading and validation from a YAML file.
*   **Dependencies:** `yaml`, `pydantic`
*   **Classes:**
    *   `ExecutionConfig(BaseModel)`
        *   `max_calls_per_prompt: int`
        *   `batch_size: int`
        *   `nshot_ct: int`
    *   `FlagsConfig(BaseModel)`
        *   `max_samples: int`
        *   `FLAG_PROMPT_PREFIX: bool`
        *   `FLAG_PROMPT_SUFFIX: bool`
        *   `prompt_prefix: str`
        *   `prompt_suffix: str`
    *   `Config(BaseModel)`
        *   `model_parameters: Dict[str, Any]`
        *   `execution: ExecutionConfig`
        *   `flags: FlagsConfig`
        *   `timeout: Dict[str, Any]`
        *   `logging: Dict[str, Any]`
        *   `output: Dict[str, Any]`
        *   `data: Dict[str, Any]`
        *   `prompts: Dict[str, str]`
        *   `model_ensemble: Dict[str, Dict[str, Any]]`
        *   `@property max_samples(self) -> int`
        *   `@property max_calls_per_prompt(self) -> int`
        *   `@property batch_size(self) -> int`
        *   `@property nshot_ct(self) -> int`
*   **Functions:**
    *   `load_config(config_file: str = "config.yaml") -> Config`
        *   Loads and validates configuration from a YAML file.

**II. `data_manager.py`**

*   **Summary:** Manages loading, preprocessing, and splitting of data.
*   **Dependencies:** `pandas`, `numpy`, `sklearn.model_selection`, `pathlib`, `typing`, `config`
*   **Classes:**
    *   `DataManager`
        *   `config: Config`
        *   `data_path: Path`
        *   `df: Optional[pd.DataFrame]`
        *   `df_train: Optional[pd.DataFrame]`
        *   `df_test: Optional[pd.DataFrame]`
        *   `df_validate: Optional[pd.DataFrame]`
        *   `_normalize_target_values(self, df: pd.DataFrame) -> pd.DataFrame`
            *   Normalizes target values in a DataFrame to 'YES' or 'NO'.
        *   `validate_data(self, df: pd.DataFrame) -> None`
            *   Validates the DataFrame for required columns and data quality.
        *   `load_and_prepare_data(self) -> Tuple[int, int]`
            *   Loads data, preprocesses it, and creates train/validate/test splits.
        *   `get_batch(self, batch_size: int, dataset: str = 'train') -> List[Dict]`
            *   Retrieves a batch of samples from the specified dataset.
        *   `get_risk_factors(self, row_id: int) -> str`
            *   Gets the risk factors for a specific row ID.
        *   `get_actual_value(self, row_id: int) -> str`
            *   Gets the actual target value for a specific row ID.
        *   `get_dataset_info(self) -> Dict[str, Any]`
            *   Returns information about the datasets.

**III. `decision.py`**

*   **Summary:** Handles interaction with the LLM API (Ollama), prompt generation, response parsing, and saving results.
*   **Dependencies:** `json`, `logging`, `typing`, `asyncio`, `datetime`, `pathlib`, `pydantic`, `ollama`, `config`, `models`, `metrics`, `utils`
*   **Classes:**
    *   `MetaData(BaseModel)`
        *   `model: Optional[str]`
        *   `created_at: Optional[str]`
        *   `done_reason: Optional[str]`
        *   `done: Optional[bool]`
        *   `total_duration: Optional[float]`
        *   `load_duration: Optional[float]`
        *   `prompt_eval_count: Optional[int]`
        *   `prompt_eval_duration: Optional[float]`
        *   `eval_count: Optional[int]`
        *   `eval_duration: Optional[float]`
*   **Functions:**
    *   `generate_output_path(model_name: str, prompt_type: PromptType, row_id: int, repeat_index: int, timestamp: str, config: Config, output_dir: Path, nshot_ct: Optional[int] = None) -> Path`
        *   Generates the output path for a decision file.
    *   `process_model_response(response_text: str) -> Dict`
        *   Parses the model's raw response text into a structured dictionary.
    *   `async get_decision(prompt_type: PromptType, model_name: str, config: Config, prompt: str) -> Tuple[Optional[Decision], Optional[MetaData], str, Dict[str, Any]]`
        *   Makes an asynchronous call to the Ollama API to get a decision.
    *   `async get_decision_with_timeout(prompt_type: PromptType, model_name: str, config: Config, prompt: str) -> Tuple[Optional[Decision], Optional[MetaData], str, Dict[str, Any], TimeoutMetrics]`
        *   Calls `get_decision` with timeout handling.
    *   `save_decision(decision: Decision, meta_data: MetaData, prompt_type: PromptType, model_name: str, row_id: int, actual_value: str, config: Config, used_prompt: str, repeat_index: int = 0, extra_data: Dict[str, Any] = None) -> bool`
        *   Saves the decision, metadata, and other relevant information to a JSON file.

**IV. `metrics.py`**

*   **Summary:** Defines data classes for storing and handling various metrics related to model evaluation.
*   **Dependencies:** `dataclasses`, `typing`
*   **Classes:**
    *   `TimeoutMetrics`
        *   `occurred: bool = False`
        *   `retry_count: int = 0`
        *   `total_timeout_duration: float = 0.0`
        *   `final_timeout_duration: Optional[float] = None`
    *   `PromptMetrics`
        *   `attempt_number: int`
        *   `execution_time_seconds: float`
        *   `successful: bool`
        *   `timeout_metrics: TimeoutMetrics`
        *   `error_message: Optional[str] = None`
        *   `prediction: Optional[str] = None`
        *   `confidence: Optional[float] = None`
        *   `meta_data: Optional[Dict[str, Any]] = None`
        *   `__post_init__(self)`
    *   `TimeoutStats`
        *   `total_timeouts: int`
        *   `avg_timeout_duration: float`
        *   `max_timeout_duration: float`
        *   `total_timeout_duration: float`

**V. `models.py`**

*   **Summary:** Defines data models (enums and Pydantic models) used throughout the application, particularly for representing risk factors, decisions, and prompt types.
*   **Dependencies:** `enum`, `typing`, `pydantic`
*   **Classes:**
    *   `PromptType(str, Enum)`
        *   `SYSTEM1 = 'system1'`
        *   `COT = 'cot'`
        *   `COT_NSHOT = 'cot-nshot'`
    *   `RiskWeight(str, Enum)`
        *   `HIGH = 'high'`
        *   `MEDIUM = 'medium'`
        *   `LOW = 'low'`
    *   `RiskFactor(BaseModel)`
        *   `factor: str`
        *   `weight: RiskWeight`
        *   `reasoning: str`
    *   `Decision(BaseModel)`
        *   `prediction: str`
        *   `confidence: int`
    *   `DecisionWithRiskFactors(Decision)`
        *   `risk_factors: List[RiskFactor]`

**VI. `performance.py`**

*   **Summary:** Handles tracking and reporting of performance metrics, including accuracy, execution time, timeout statistics, and more.
*   **Dependencies:** `dataclasses`, `datetime`, `typing`, `statistics`, `logging`, `json`, `pandas`, `pathlib`, `sklearn.metrics`, `metrics`
*   **Classes:**
    *   `PerformanceStats`
        *   `prompt_type: str`
        *   `model_name: str`
        *   `start_time: datetime`
        *   `end_time: datetime`
        *   `total_attempts: int`
        *   `successful_attempts: int`
        *   `failed_attempts: int`
        *   `timeout_attempts: int`
        *   `avg_execution_time: float`
        *   `median_execution_time: float`
        *   `sd_execution_time: float`
        *   `timeout_stats: TimeoutStats`
        *   `meta_data_averages: Dict[str, float]`
        *   `meta_data_sd: Dict[str, float]`
        *   `prediction_accuracy: float = 0.0`
        *   `prediction_distribution: Dict[str, int] = None`
        *   `actual_distribution: Dict[str, int] = None`
        *   `confusion_matrix: Dict[str, int] = None`
        *   `auc_roc: float = 0.0`
    *   `DecisionTracker`
        *   `total_predictions: int`
        *   `correct_predictions: int`
        *   `actual_values: List[str]`
        *   `predicted_values: List[str]`
        *   `confidences: List[Optional[float]]`
        *   `record_prediction(self, prediction: str, actual: str, confidence: Optional[float] = None)`
            *   Records a single prediction.
        *   `get_accuracy(self) -> float`
            *   Calculates the prediction accuracy.
        *   `get_stats(self) -> Dict[str, Any]`
            *   Returns comprehensive prediction statistics.
    *   `PerformanceTracker`
        *   `prompt_type: str`
        *   `model_name: str`
        *   `metrics: List[PromptMetrics]`
        *   `attempts: List[PromptMetrics]`
        *   `start_time: datetime`
        *   `decision_tracker: DecisionTracker`
        *   `record_attempt(self, metrics: PromptMetrics)`
            *   Records the results from a single prompt execution attempt.
        *   `_calculate_timeout_stats(self) -> TimeoutStats`
            *   Computes aggregated statistics about timeouts.
        *   `_calculate_meta_data_stats(self) -> Tuple[Dict[str, float], Dict[str, float]]`
            *   Calculates averages and standard deviations for numeric metadata.
        *   `_generate_stats(self) -> Optional[PerformanceStats]`
            *   Computes final `PerformanceStats` from all recorded attempts.
        *   `save_metrics(self, execution_time: float)`
            *   Saves the performance statistics to a JSON file and a text report.
        *   `_save_text_report(self, stats: PerformanceStats, execution_time: float, timestamp: str, output_dir: Path)`
            *   Generates a readable text report of the performance statistics.
*   **Functions:**
    *   `save_aggregate_stats(session_results: List[PerformanceStats], total_duration: float)`
        *   Summarizes multiple performance tracking sessions into an aggregate report.

**VII. `prompt_manager.py`**

*   **Summary:** Manages the generation of dynamic prompts based on configuration and data.
*   **Dependencies:** `logging`, `typing`, `random`, `config`, `data_manager`, `models`
*   **Classes:**
    *   `PromptManager`
        *   `config: Config`
        *   `data_manager: DataManager`
        *   `_base_prompts: Dict[PromptType, str]`
        *   `_nshot_examples: Optional[str]`
        *   `_generate_nshot_examples(self) -> str`
            *   Generates n-shot examples for the `COT_NSHOT` prompt type.
        *   `get_prompt(self, prompt_type: PromptType, row_id: int) -> str`
            *   Generates a prompt with dynamically inserted risk factors and n-shot examples (if applicable).

**VIII. `utils.py`**

*   **Summary:** Provides utility functions used across the project.
*   **Dependencies:** `typing`, `pathlib`
*   **Functions:**
    *   `clean_model_name(model_name: str) -> str`
        *   Cleans a model name for safe file system usage.
    *   `pydantic_or_dict(obj: Any) -> Any`
        *   Converts Pydantic objects to dictionaries, or returns the object if not Pydantic.
    *   `convert_ns_to_s(data: Any) -> Any`
        *   Converts nanosecond values to seconds in nested dictionaries.
    *   `check_existing_decision(output_path: pathlib.Path) -> bool`
        *   Checks if a decision file already exists at the given path.

### 2. Overall Code Architecture, Control Flow, Design Choices, and Improvements

**Architecture:**

The code follows a modular design, separating concerns into different files/modules:

*   **Configuration:** `config.py` handles loading and validating configuration from a YAML file.
*   **Data Management:** `data_manager.py` is responsible for loading, preprocessing, and splitting the dataset.
*   **Prompt Generation:** `prompt_manager.py` dynamically creates prompts for the LLM, including n-shot examples.
*   **LLM Interaction:** `decision.py` interacts with the Ollama API, sends prompts, receives responses, parses the responses, and saves the results.
*   **Metrics and Reporting:** `metrics.py` and `performance.py` track, calculate, and report various performance metrics.
*   **Models:** `models.py` defines data structures (enums and Pydantic models) for type hinting and data validation.
*   **Utilities:** `utils.py` contains helper functions used across the project.

**Control Flow:**

1.  **Initialization:**
    *   The `main.py` script (which would be the entry point) would initialize the `Config`, `DataManager`, and `PromptManager`.
    *   The `DataManager` loads and preprocesses the data.
2.  **Prompt Generation and Model Evaluation Loop:**
    *   The main script would iterate through a batch of data (or the entire dataset).
    *   For each data sample:
        *   `PromptManager` generates a prompt based on the configured prompt type and the sample's risk factors.
        *   `decision.py` sends the prompt to the Ollama API using `get_decision_with_timeout`.
        *   The response is parsed and validated.
        *   The decision and metadata are saved to a JSON file using `save_decision`.
        *   `PerformanceTracker` records the attempt's metrics.
3.  **Performance Reporting:**
    *   After processing all samples, `PerformanceTracker` generates a report (both JSON and human-readable text) summarizing the model's performance.
    *   `save_aggregate_stats` can be used to create an aggregate report across multiple evaluation sessions.

**Design Choices and Tradeoffs:**

*   **Modularity:** The code is well-modularized, making it easier to maintain, extend, and test individual components.
*   **Configuration-Driven:** Using a YAML configuration file allows for easy modification of parameters without changing the code.
*   **Type Hinting:** Extensive use of type hinting (with `typing` and Pydantic) improves code readability and helps catch errors early.
*   **Error Handling:** The code includes error handling (e.g., in `data_manager.py`, `decision.py`) to gracefully handle issues like invalid data, API errors, and timeouts.
*   **Asynchronous Operations:** Using `asyncio` for API calls allows for concurrent processing of multiple prompts, potentially improving performance.
*   **Timeout Handling:** `get_decision_with_timeout` provides a mechanism to handle API calls that take too long.
*   **Ollama API:** The code is designed to work specifically with the Ollama API for local LLM inference.
*   **Tradeoff: Complexity:** The extensive use of classes and object-oriented design might make the code slightly more complex for beginners.
*   **Tradeoff: Ollama Dependency:** The code is tightly coupled to the Ollama API. Porting it to another LLM API would require modifications.

**Possible Future Improvements:**

*   **Abstract LLM API Interaction:** Create an abstract base class or interface for LLM interaction. This would allow for easier switching between different