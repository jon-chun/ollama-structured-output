# utils.py
import asyncio
import logging
from typing import Any, Callable, TypeVar, Coroutine
from datetime import datetime
from pathlib import Path
import json

T = TypeVar('T')

class AsyncFileHandler:
    """Handles asynchronous file operations with proper error handling"""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path

    async def write_json(self, data: Any):
        """Writes data to a JSON file asynchronously"""
        try:
            json_str = json.dumps(data, indent=2, default=str)
            async with aopen(self.file_path, 'w') as f:
                await f.write(json_str)
        except Exception as e:
            logging.error(f"Error writing JSON to {self.file_path}: {e}")
            raise

    async def read_json(self) -> Any:
        """Reads JSON data from a file asynchronously"""
        try:
            async with aopen(self.file_path, 'r') as f:
                content = await f.read()
            return json.loads(content)
        except Exception as e:
            logging.error(f"Error reading JSON from {self.file_path}: {e}")
            raise

class RetryHandler:
    """Implements retry logic with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def retry_with_backoff(
        self,
        operation: Callable[..., Coroutine[Any, Any, T]],
        *args,
        **kwargs
    ) -> T:
        """
        Executes an operation with retry logic and exponential backoff
        
        Args:
            operation: Async function to retry
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation if successful
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    logging.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logging.error(
                        f"All retry attempts failed for operation: {operation.__name__}"
                    )
        
        raise last_exception

class MetricsFormatter:
    """Formats metrics data for various output formats"""
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Formats a duration in seconds to a human-readable string"""
        if seconds < 60:
            return f"{seconds:.2f}s"
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}m {remaining_seconds:.2f}s"

# utils.py (continued)
    @staticmethod
    def format_percentage(value: float, decimal_places: int = 2) -> str:
        """Formats a decimal value as a percentage with specified precision"""
        return f"{value:.{decimal_places}f}%"

    @staticmethod
    def format_token_count(count: int) -> str:
        """Formats token counts with thousand separators"""
        return f"{count:,}"

    @staticmethod
    def format_timestamp(dt: datetime) -> str:
        """Formats timestamps in a consistent, readable format"""
        return dt.strftime("%Y-%m-%d %H:%M:%S")

class TimeoutCalculator:
    """Calculates and manages timeout durations with exponential backoff"""
    
    def __init__(self, base_timeout: float, max_timeout: float, backoff_factor: float = 1.5):
        self.base_timeout = base_timeout
        self.max_timeout = max_timeout
        self.backoff_factor = backoff_factor

    def calculate_timeout(self, attempt: int) -> float:
        """
        Calculates timeout duration for a given attempt number
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Timeout duration in seconds, capped at max_timeout
        """
        timeout = self.base_timeout * (self.backoff_factor ** attempt)
        return min(timeout, self.max_timeout)

class ModelMetricsCollector:
    """Collects and processes model-specific performance metrics"""

    def __init__(self):
        self.success_counts = defaultdict(int)
        self.total_counts = defaultdict(int)
        self.execution_times = defaultdict(list)
        self.token_counts = defaultdict(list)

    def record_attempt(
        self,
        model_name: str,
        success: bool,
        execution_time: float,
        token_count: int
    ):
        """Records metrics for a single model attempt"""
        self.total_counts[model_name] += 1
        if success:
            self.success_counts[model_name] += 1
        self.execution_times[model_name].append(execution_time)
        self.token_counts[model_name].append(token_count)

    def get_model_metrics(self, model_name: str) -> Dict[str, float]:
        """
        Retrieves aggregated metrics for a specific model
        
        Returns:
            Dictionary containing:
            - success_rate: Percentage of successful attempts
            - avg_execution_time: Average execution time in seconds
            - avg_token_count: Average number of tokens per attempt
        """
        total = self.total_counts[model_name]
        if total == 0:
            return {
                'success_rate': 0.0,
                'avg_execution_time': 0.0,
                'avg_token_count': 0.0
            }

        return {
            'success_rate': (self.success_counts[model_name] / total) * 100,
            'avg_execution_time': mean(self.execution_times[model_name]),
            'avg_token_count': mean(self.token_counts[model_name])
        }

class AsyncContextTimer:
    """Context manager for timing async operations with logging"""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time: Optional[float] = None
        self.duration: Optional[float] = None

    async def __aenter__(self) -> 'AsyncContextTimer':
        """Enters the context and starts the timer"""
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.operation_name}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exits the context and logs the duration"""
        if self.start_time is not None:
            self.duration = time.time() - self.start_time
            self.logger.debug(
                f"Completed {self.operation_name} in "
                f"{MetricsFormatter.format_duration(self.duration)}"
            )

class ConfigValidator:
    """Validates configuration settings with detailed error messages"""
    
    @staticmethod
    def validate_timeout_config(config: Dict[str, Any]) -> bool:
        """
        Validates timeout-related configuration settings
        
        Args:
            config: Dictionary containing timeout configuration
            
        Returns:
            True if valid, raises ConfigurationError if invalid
        """
        required_fields = {
            'max_api_wait_sec': (float, int),
            'max_api_timeout_retries': int,
            'api_wait_step_increase_sec': (float, int)
        }

        for field, expected_type in required_fields.items():
            if field not in config:
                raise ConfigurationError(f"Missing required timeout config: {field}")
            
            value = config[field]
            if not isinstance(value, expected_type):
                raise ConfigurationError(
                    f"Invalid type for {field}: expected {expected_type}, "
                    f"got {type(value)}"
                )
            
            if isinstance(value, (float, int)) and value <= 0:
                raise ConfigurationError(f"{field} must be positive")

        return True

    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> bool:
        """
        Validates model-specific configuration settings
        
        Args:
            config: Dictionary containing model configuration
            
        Returns:
            True if valid, raises ConfigurationError if invalid
        """
        required_fields = {
            'model_temperature': (float, 0.0, 1.0),
            'model_top_p': (float, 0.0, 1.0),
            'model_max_tokens': (int, 1, None)
        }

        for field, (expected_type, min_val, max_val) in required_fields.items():
            if field not in config:
                raise ConfigurationError(f"Missing required model config: {field}")
            
            value = config[field]
            if not isinstance(value, expected_type):
                raise ConfigurationError(
                    f"Invalid type for {field}: expected {expected_type}, "
                    f"got {type(value)}"
                )
            
            if min_val is not None and value < min_val:
                raise ConfigurationError(f"{field} must be >= {min_val}")
            
            if max_val is not None and value > max_val:
                raise ConfigurationError(f"{field} must be <= {max_val}")

        return True

async def safe_file_operation(operation: Callable, *args, **kwargs) -> Any:
    """
    Executes file operations safely with proper error handling
    
    Args:
        operation: File operation to perform
        *args: Positional arguments for the operation
        **kwargs: Keyword arguments for the operation
        
    Returns:
        Result of the operation if successful
        
    Raises:
        IOError: If file operation fails
    """
    try:
        if asyncio.iscoroutinefunction(operation):
            return await operation(*args, **kwargs)
        return operation(*args, **kwargs)
    except Exception as e:
        logging.error(f"File operation failed: {str(e)}")
        raise IOError(f"File operation failed: {str(e)}") from e