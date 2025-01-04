# Add new imports at the top
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

# Constants for data processing
SPLIT_PERCENT_TRAIN = 70
RANDOM_SEED = 42

class DataManager:
    """Manages data loading, splitting, and preprocessing for model evaluation"""
    def __init__(self, data_path: str = os.path.join('.', 'data', 'vignettes_renamed_clean.csv')):
        self.data_path = data_path
        self.df = None
        self.df_train = None
        self.df_test = None
        
    def load_and_prepare_data(self) -> None:
        """Load CSV data and prepare train/test splits"""
        # Read CSV and add sequence ID
        self.df = pd.read_csv(self.data_path)
        self.df['id'] = np.arange(len(self.df))
        
        # Set random seed and shuffle
        np.random.seed(RANDOM_SEED)
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        
        # Split into train/test sets
        train_size = SPLIT_PERCENT_TRAIN / 100.0
        self.df_train, self.df_test = train_test_split(
            self.df, 
            train_size=train_size,
            random_state=RANDOM_SEED
        )
        
        logging.info(f"Loaded {len(self.df)} total samples")
        logging.info(f"Split into {len(self.df_train)} train and {len(self.df_test)} test samples")

class PromptManager:
    """Enhanced prompt manager with dynamic risk factor generation"""
    def __init__(self, config: Config, data_manager: DataManager):
        self.config = config
        self.data_manager = data_manager
        self._base_prompts = {
            PromptType.SYSTEM1: config.prompts["system1"],
            PromptType.COT: config.prompts["cot"]
        }
    
    def get_prompt(self, prompt_type: PromptType, row_id: int) -> str:
        """Get prompt with dynamically inserted risk factors from data"""
        base_prompt = self._base_prompts[prompt_type]
        
        # Get risk factors text from training data
        risk_factors = self.data_manager.df_train.loc[
            self.data_manager.df_train['id'] == row_id, 
            'short_text_summary'
        ].iloc[0]
        
        # Replace placeholder with actual risk factors
        return base_prompt.replace('###RISK_FACTORS:', risk_factors)

class DecisionTracker:
    """Tracks and accumulates prediction statistics"""
    def __init__(self):
        self.total_predictions = 0
        self.correct_predictions = 0
        self.actual_values = []
        self.predicted_values = []
        
    def record_prediction(self, prediction: str, actual: str):
        """Record a single prediction and its actual value"""
        self.total_predictions += 1
        is_correct = prediction.upper() == actual.upper()
        if is_correct:
            self.correct_predictions += 1
        
        self.actual_values.append(actual)
        self.predicted_values.append(prediction)
    
    def get_accuracy(self) -> float:
        """Calculate prediction accuracy"""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions * 100

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive prediction statistics"""
        return {
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'accuracy': self.get_accuracy(),
            'actual_distribution': pd.Series(self.actual_values).value_counts().to_dict(),
            'predicted_distribution': pd.Series(self.predicted_values).value_counts().to_dict()
        }

def save_decision(
    decision: Decision,
    meta_data: MetaData,
    prompt_type: PromptType,
    model_name: str,
    row_id: int,
    actual_value: str,
    config: Config
) -> bool:
    """Enhanced save_decision function with additional metadata"""
    try:
        output_dir = Path(config.output["base_dir"]) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with row ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{model_name}_{prompt_type}_id{row_id}_{timestamp}.json"
        
        # Convert to serializable format
        decision_data = pydantic_or_dict(decision)
        meta_data_data = pydantic_or_dict(meta_data)
        
        # Add additional fields
        decision_data.update({
            'id': row_id,
            'actual': actual_value,
            'correct': "YES" if str(decision.prediction).upper() == actual_value.upper() else "NO"
        })
        
        # Convert timestamps from ns to s
        convert_ns_to_s(meta_data_data)
        
        combined_data = {
            "decision": decision_data,
            "meta_data": meta_data_data
        }
        
        output_path = output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, default=str)
        
        logging.info(f"Successfully saved decision+meta_data to {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error saving decision: {str(e)}")
        return False

class PerformanceTracker:
    """Enhanced performance tracker with prediction statistics"""
    def __init__(self, prompt_type: str, model_name: str):
        self.prompt_type = prompt_type
        self.model_name = model_name
        self.metrics: List[PromptMetrics] = []
        self.start_time = datetime.now()
        self.decision_tracker = DecisionTracker()

    def save_metrics(self, execution_time: float):
        """Enhanced metrics saving with prediction statistics"""
        stats = self._generate_stats()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add prediction statistics
        prediction_stats = self.decision_tracker.get_stats()
        
        # Save JSON format
        json_path = f"metrics_{self.model_name}_{self.prompt_type}_{timestamp}.json"
        with open(json_path, 'w') as f:
            stats_dict = asdict(stats)
            stats_dict['prediction_stats'] = prediction_stats
            json.dump(stats_dict, f, indent=2, default=str)
        
        # Save text report with prediction statistics
        self._save_text_report(stats, execution_time, timestamp, prediction_stats)

    def _save_text_report(self, stats: PerformanceStats, execution_time: float, 
                         timestamp: str, prediction_stats: Dict[str, Any]):
        """Enhanced text report with prediction statistics"""
        report_path = f"report_{stats.model_name}_{stats.prompt_type}_{timestamp}.txt"
        with open(report_path, 'w') as f:
            # ... [Previous report content] ...
            
            f.write("\nPrediction Statistics\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Predictions: {prediction_stats['total_predictions']}\n")
            f.write(f"Correct Predictions: {prediction_stats['correct_predictions']}\n")
            f.write(f"Accuracy: {prediction_stats['accuracy']:.2f}%\n\n")
            
            f.write("Actual Value Distribution:\n")
            for value, count in prediction_stats['actual_distribution'].items():
                f.write(f"  {value}: {count}\n")
            
            f.write("\nPredicted Value Distribution:\n")
            for value, count in prediction_stats['predicted_distribution'].items():
                f.write(f"  {value}: {count}\n")

async def run_evaluation_cycle(
    model_name: str,
    prompt_type: PromptType,
    config: Config,
    tracker: PerformanceTracker,
    data_manager: DataManager,
    prompt_manager: PromptManager
) -> None:
    """Enhanced evaluation cycle with data processing"""
    for _, row in data_manager.df_train.iterrows():
        start_time = time.time()
        row_id = row['id']
        actual_value = row['target']
        
        logging.info(
            f"Processing row {row_id} for model={model_name}, "
            f"prompt_type={prompt_type}"
        )
        
        try:
            # Get dynamic prompt for this row
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
                # Record prediction
                tracker.decision_tracker.record_prediction(
                    str(decision.prediction),
                    actual_value
                )
                
                # Save decision with additional metadata
                save_success = save_decision(
                    decision, meta_data, prompt_type, model_name,
                    row_id, actual_value, config
                )
                
                if not save_success:
                    logging.warning("Decision valid but save failed")
                
                # Record metrics
                metrics = PromptMetrics(
                    attempt_number=row_id + 1,
                    execution_time_seconds=execution_time,
                    successful=True,
                    timeout_metrics=timeout_metrics,
                    prediction=str(decision.prediction),
                    confidence=float(decision.confidence),
                    meta_data=meta_data
                )
                
                # Display results
                print(f"\nModel: {model_name}")
                print(f"Prompt Type: {prompt_type}")
                print(f"Row ID: {row_id}")
                print(f"Prediction: {decision.prediction}")
                print(f"Actual: {actual_value}")
                print(f"Correct: {'YES' if str(decision.prediction).upper() == actual_value.upper() else 'NO'}")
                print(f"Confidence: {decision.confidence}%")
                
            else:
                # Handle failed decision
                metrics = PromptMetrics(
                    attempt_number=row_id + 1,
                    execution_time_seconds=execution_time,
                    successful=False,
                    timeout_metrics=timeout_metrics,
                    error_message="No valid decision received"
                )
            
            # Record metrics
            tracker.record_attempt(metrics)
            
        except Exception as e:
            logging.error(f"Error processing row {row_id}: {str(e)}")
            continue

async def main():
    """Enhanced main function with data processing"""
    config = Config()
    overall_start = time.time()
    
    # Initialize logging
    logging.basicConfig(
        level=getattr(logging, config.logging["level"]),
        format=config.logging["format"],
        handlers=[
            logging.FileHandler(config.logging["file"]),
            logging.StreamHandler()
        ]
    )
    
    # Initialize data management
    data_manager = DataManager()
    try:
        data_manager.load_and_prepare_data()
    except Exception as e:
        logging.error(f"Failed to load data: {str(e)}")
        return
    
    # Create output directory
    output_base = Path(config.output["base_dir"])
    output_base.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize prompt manager
        prompt_manager = PromptManager(config, data_manager)
        
        # Track all session results
        session_results: List[PerformanceStats] = []
        
        # Evaluate each model
        for model_name, model_config in config.model_ensemble.items():
            logging.info(f"Starting evaluation of model: {model_name}")
            model_start = time.time()
            
            # Test each prompt type
            for prompt_type in PromptType:
                logging.info(f"Testing prompt type: {prompt_type}")
                
                try:
                    # Initialize tracker
                    tracker = PerformanceTracker(prompt_type, model_name)
                    
                    # Run evaluation cycle
                    await run_evaluation_cycle(
                        model_name, prompt_type, config, tracker,
                        data_manager, prompt_manager
                    )
                    
                    # Generate and save statistics
                    session_stats = tracker._generate_stats()
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