# LLM Decisions Project Technical Documentation

## 1. Code Structure and Components

### 1.1 Core Modules

#### config.py
- **Primary Classes:**
  - `ExecutionConfig(BaseModel)`
    - Properties: max_calls_per_prompt: int, batch_size: int, nshot_ct: int
  - `FlagsConfig(BaseModel)` 
    - Properties: max_samples: int, FLAG_PROMPT_PREFIX: bool, FLAG_PROMPT_SUFFIX: bool, prompt_prefix: str, prompt_suffix: str
  - `Config(BaseModel)`
    - Properties: model_parameters: Dict, execution: ExecutionConfig, flags: FlagsConfig, timeout: Dict, logging: Dict, output: Dict, data: Dict, prompts: Dict, model_ensemble: Dict
    - Methods: max_samples(), max_calls_per_prompt(), batch_size(), nshot_ct()

Functions:
- `load_config(config_file: str) -> Config`
  - Loads YAML configuration file and validates against schema

#### models.py
- **Enums:**
  - `PromptType`: SYSTEM1, COT, COT_NSHOT
  - `RiskWeight`: HIGH, MEDIUM, LOW

- **Models:**
  - `RiskFactor(BaseModel)`
    - Properties: factor: str, weight: RiskWeight, reasoning: str
  - `Decision(BaseModel)` 
    - Properties: prediction: str, confidence: int
  - `DecisionWithRiskFactors(Decision)`
    - Properties: risk_factors: List[RiskFactor]

#### data_manager.py
- **Primary Class:** 
  - `DataManager`
    - Properties: config: Config, data_path: Path, df/df_train/df_test/df_validate: Optional[pd.DataFrame]
    - Methods:
      - load_and_prepare_data() -> Tuple[int, int]
      - get_batch(batch_size: int, dataset: str) -> List[Dict]
      - get_risk_factors(row_id: int) -> str
      - get_actual_value(row_id: int) -> str
      - get_dataset_info() -> Dict[str, Any]

#### decision.py
- **Models:**
  - `MetaData(BaseModel)`
    - Properties: Various optional fields for API response metadata

Functions:
- `generate_output_path(...) -> Path`
- `process_model_response(response_text: str) -> Dict`
- `get_decision(...) -> Tuple[Optional[Decision], Optional[MetaData], str, Dict[str, Any]]`
- `get_decision_with_timeout(...) -> Tuple[...]`
- `save_decision(...) -> bool`

#### metrics.py
- **DataClasses:**
  - `TimeoutMetrics`
  - `PromptMetrics`
  - `TimeoutStats`

#### performance.py
- **DataClasses:**
  - `PerformanceStats`

- **Classes:**
  - `DecisionTracker`
    - Methods: record_prediction(), get_accuracy(), get_stats()
  - `PerformanceTracker`
    - Methods: record_attempt(), save_metrics(), _save_text_report()

#### prompt_manager.py
- **Primary Class:**
  - `PromptManager`
    - Properties: config: Config, data_manager: DataManager
    - Methods:
      - _generate_nshot_examples() -> str
      - get_prompt(prompt_type: PromptType, row_id: int) -> str

## 2. Architecture Overview

### 2.1 Design Pattern
The project follows a modular architecture with clear separation of concerns:

1. **Configuration Management**
   - Centralized configuration through YAML
   - Strong validation using Pydantic models
   - Hierarchical configuration structure

2. **Data Pipeline**
   - Robust data loading and preprocessing
   - Train/validation/test splitting with stratification
   - Batch processing capabilities

3. **Model Integration**
   - Asynchronous model interaction
   - Timeout handling and retry mechanisms
   - Standardized response processing

4. **Metrics and Performance Tracking**
   - Comprehensive metrics collection
   - Performance analysis and reporting
   - Structured output generation

### 2.2 Control Flow

1. **Initialization**
   ```
   Config Loading -> Data Manager Setup -> Prompt Manager Setup
   ```

2. **Execution Pipeline**
   ```
   Data Loading -> Batch Processing -> Model Inference -> Results Collection
   ```

3. **Results Processing**
   ```
   Metrics Collection -> Performance Analysis -> Report Generation
   ```

### 2.3 Key Design Choices

1. **Async Processing**
   - Use of asyncio for model interactions
   - Timeout handling with retry mechanisms
   - Batch processing capability

2. **Data Management**
   - Pandas for efficient data handling
   - Stratified splitting for balanced datasets
   - Flexible batch processing

3. **Configuration**
   - Strong typing with Pydantic
   - YAML-based configuration
   - Hierarchical structure

4. **Error Handling**
   - Comprehensive logging
   - Graceful failure handling
   - Detailed error reporting

### 2.4 Strong Points

1. **Modularity**
   - Clear separation of concerns
   - Easy to extend and modify
   - Well-defined interfaces

2. **Robustness**
   - Strong type checking
   - Comprehensive error handling
   - Detailed logging

3. **Scalability**
   - Batch processing
   - Async capabilities
   - Configurable parameters

## 3. Potential Improvements

### 3.1 Technical Improvements

1. **Parallelization**
   - Implement parallel batch processing
   - Add multi-model concurrent execution
   - Optimize data loading for large datasets

2. **Caching**
   - Add result caching mechanism
   - Implement prompt template caching
   - Add model response caching

3. **Monitoring**
   - Add real-time monitoring
   - Implement progress tracking
   - Add performance profiling

### 3.2 Feature Additions

1. **Model Management**
   - Model versioning system
   - Model performance comparison
   - Automatic model selection

2. **Data Processing**
   - Advanced preprocessing options
   - Custom splitting strategies
   - Data augmentation capabilities

3. **Reporting**
   - Interactive visualizations
   - Automated report generation
   - Custom metric definitions

### 3.3 Infrastructure

1. **Containerization**
   - Docker support
   - Kubernetes deployment configs
   - CI/CD pipeline integration

2. **API Layer**
   - REST API interface
   - WebSocket support
   - API documentation

3. **Storage**
   - Database integration
   - Cloud storage support
   - Caching layer

## 4. Development Guidelines

### 4.1 Code Style

1. **Type Hints**
   - Use type hints consistently
   - Leverage Pydantic for validation
   - Document complex types

2. **Documentation**
   - Maintain docstrings
   - Update README
   - Keep comments current

3. **Testing**
   - Add unit tests
   - Implement integration tests
   - Add performance tests

### 4.2 Best Practices

1. **Error Handling**
   - Use custom exceptions
   - Implement proper logging
   - Add error recovery

2. **Configuration**
   - Use environment variables
   - Implement secrets management
   - Add configuration validation

3. **Performance**
   - Profile code regularly
   - Optimize critical paths
   - Monitor resource usage

## 5. Conclusion

The codebase demonstrates a well-structured approach to LLM decision evaluation with strong emphasis on reliability and extensibility. The modular design allows for easy maintenance and future improvements. Key areas for enhancement include parallelization, monitoring, and additional feature implementation.
