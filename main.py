# main.py
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# Local imports
from config import Config
from data_manager import DataManager
from prompt_manager import PromptManager
from models import PromptType
from metrics import TimeoutMetrics, PromptMetrics
from performance import (
    PerformanceTracker,
    PerformanceStats,
    save_aggregate_stats
)
from decision import (
    Decision,
    MetaData,
    get_decision_with_timeout,
    save_decision
)

async def run_evaluation_cycle(
    model_name: str,
    prompt_type: PromptType,
    config: Config,
    tracker: PerformanceTracker,
    data_manager: DataManager,
    prompt_manager: PromptManager
) -> None:
    """
    Run an evaluation cycle for a specific model and prompt type combination.
    
    This function processes batches of data from the training set, making predictions
    and tracking performance metrics throughout the evaluation. It handles batch
    processing efficiently while maintaining detailed logging and error handling.
    
    Args:
        model_name: Name of the model being evaluated
        prompt_type: Type of prompt being used
        config: Configuration settings
        tracker: Performance tracking instance
        data_manager: Data management instance
        prompt_manager: Prompt management instance
    """
    # Get batch size from config or use default
    batch_size = config.execution.get("batch_size", 100)
    
    # Get dataset information for logging
    dataset_info = data_manager.get_dataset_info()
    logging.info(
        f"Starting evaluation cycle for {model_name} with {prompt_type}\n"
        f"Dataset size: {dataset_info['dataset_sizes']['train']} samples\n"
        f"Target distribution: {dataset_info['target_distributions']['train']}"
    )
    
    # Get all training data as batches
    remaining_samples = dataset_info['dataset_sizes']['train']
    processed_samples = 0
    
    while remaining_samples > 0:
        # Calculate current batch size
        current_batch_size = min(batch_size, remaining_samples)
        
        try:
            # Get batch of data
            data_batch = data_manager.get_batch(current_batch_size)
            
            # Process each sample in the batch
            for sample in data_batch:
                start_time = time.time()
                row_id = sample['id']
                actual_value = sample['target']
                
                logging.info(
                    f"Processing sample {processed_samples + 1}/{dataset_info['dataset_sizes']['train']} "
                    f"(ID: {row_id})"
                )
                
                try:
                    # Get dynamic prompt for this sample
                    prompt = prompt_manager.get_prompt(prompt_type, row_id)
                    
                    # Get decision with timeout handling
                    decision, meta_data, timeout_metrics = await get_decision_with_timeout(
                        prompt_type=prompt_type,
                        model_name=model_name,
                        config=config,
                        prompt=prompt
                    )
                    
                    execution_time = time.time() - start_time
                    
                    if decision is not None:
                        # Record prediction in the tracker
                        tracker.decision_tracker.record_prediction(
                            str(decision.prediction),
                            actual_value
                        )
                        
                        # Save decision with additional metadata
                        save_success = save_decision(
                            decision=decision,
                            meta_data=meta_data,
                            prompt_type=prompt_type,
                            model_name=model_name,
                            row_id=row_id,
                            actual_value=actual_value,
                            config=config
                        )
                        
                        if not save_success:
                            logging.warning("Decision valid but save failed")
                        
                        # Create and record metrics
                        metrics = PromptMetrics(
                            attempt_number=processed_samples + 1,
                            execution_time_seconds=execution_time,
                            successful=True,
                            timeout_metrics=timeout_metrics,
                            prediction=str(decision.prediction),
                            confidence=float(decision.confidence),
                            meta_data=meta_data
                        )
                        
                        # Display detailed results
                        print(f"\nEvaluation Results for Sample {processed_samples + 1}")
                        print("-" * 40)
                        print(f"Model: {model_name}")
                        print(f"Prompt Type: {prompt_type}")
                        print(f"Sample ID: {row_id}")
                        print(f"Prediction: {decision.prediction}")
                        print(f"Confidence: {decision.confidence}%")
                        print(f"Actual Value: {actual_value}")
                        print(f"Correct: {'YES' if str(decision.prediction).upper() == actual_value.upper() else 'NO'}")
                        print(f"Execution Time: {execution_time:.2f}s")
                        if timeout_metrics.occurred:
                            print(f"Timeout Retries: {timeout_metrics.retry_count}")
                        print("-" * 40)
                        
                    else:
                        # Handle failed prediction
                        metrics = PromptMetrics(
                            attempt_number=processed_samples + 1,
                            execution_time_seconds=execution_time,
                            successful=False,
                            timeout_metrics=timeout_metrics,
                            error_message="No valid decision received"
                        )
                    
                    # Record metrics for this attempt
                    tracker.record_attempt(metrics)
                    
                except Exception as e:
                    logging.error(f"Error processing sample {row_id}: {str(e)}")
                    continue
                
                processed_samples += 1
                
            remaining_samples -= current_batch_size
            
            # Log batch completion
            if remaining_samples > 0:
                logging.info(
                    f"Completed batch. Remaining samples: {remaining_samples}"
                )
            
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            remaining_samples = 0  # Stop processing on batch error

async def run_evaluation_session(
    model_name: str,
    prompt_type: PromptType,
    config: Config,
    data_manager: DataManager,
    prompt_manager: PromptManager
) -> Optional[PerformanceStats]:
    """
    Run a complete evaluation session for a model/prompt combination.
    
    This function manages the evaluation process for a specific model and prompt type,
    including performance tracking and statistics generation.
    
    Args:
        model_name: Name of the model to evaluate
        prompt_type: Type of prompt to use
        config: Configuration settings
        data_manager: Data management instance
        prompt_manager: Prompt management instance
        
    Returns:
        PerformanceStats if successful, None if critical failure
    """
    # Initialize performance tracking
    tracker = PerformanceTracker(prompt_type, model_name)
    session_start = time.time()
    
    try:
        # Run the evaluation cycle
        await run_evaluation_cycle(
            model_name=model_name,
            prompt_type=prompt_type,
            config=config,
            tracker=tracker,
            data_manager=data_manager,
            prompt_manager=prompt_manager
        )
        
        # Calculate session duration and save metrics
        session_duration = time.time() - session_start
        tracker.save_metrics(session_duration)
        
        # Generate and return statistics
        return tracker._generate_stats()
        
    except Exception as e:
        logging.error(f"Critical error in evaluation session: {str(e)}")
        return None

async def main():
    """
    Main execution function for the evaluation system.
    
    This function orchestrates the entire evaluation process, including:
    - Configuration loading
    - Data preparation
    - Model evaluation
    - Performance tracking
    - Results aggregation
    """
    # Initialize configuration and timing
    config = Config()
    overall_start = time.time()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config.logging["level"]),
        format=config.logging["format"],
        handlers=[
            logging.FileHandler(config.logging["file"]),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Initialize data management
        data_manager = DataManager(config)
        data_manager.load_and_prepare_data()
        
        # Create output directory
        output_base = Path(config.output["base_dir"])
        output_base.mkdir(parents=True, exist_ok=True)
        
        # Initialize prompt manager
        prompt_manager = PromptManager(config, data_manager)
        
        # Track session results
        session_results: List[PerformanceStats] = []
        
        # Evaluate each model
        for model_name, model_config in config.model_ensemble.items():
            logging.info(f"Starting evaluation of model: {model_name}")
            model_start = time.time()
            
            # Test each prompt type
            for prompt_type in PromptType:
                logging.info(f"Testing prompt type: {prompt_type}")
                
                try:
                    # Run evaluation session
                    session_stats = await run_evaluation_session(
                        model_name=model_name,
                        prompt_type=prompt_type,
                        config=config,
                        data_manager=data_manager,
                        prompt_manager=prompt_manager
                    )
                    
                    if session_stats:
                        session_results.append(session_stats)
                        
                except Exception as e:
                    logging.error(f"Error in session {model_name}/{prompt_type}: {str(e)}")
                    continue
            
            # Log model completion
            model_duration = time.time() - model_start
            logging.info(
                f"Completed evaluation of {model_name} "
                f"in {model_duration:.2f}s"
            )
        
        # Generate and save aggregate statistics
        total_duration = time.time() - overall_start
        if session_results:
            try:
                save_aggregate_stats(session_results, total_duration)
            except Exception as e:
                logging.error(f"Error saving aggregate stats: {str(e)}")
        else:
            logging.warning("No session results to aggregate")
        
        logging.info(f"Completed all evaluations in {total_duration:.2f}s")
        
    except Exception as e:
        logging.error(f"Critical error in main execution: {str(e)}")
        raise
    
    finally:
        logging.info("Evaluation process finished")

if __name__ == "__main__":
    asyncio.run(main())