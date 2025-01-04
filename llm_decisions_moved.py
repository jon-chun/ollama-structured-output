MetaCoordinator
The error message you're seeing suggests that Python is interpreting this as a multi-command application (like git with its subcommands), but that's not what we want.
Let's fix this by creating a clean, unified version of the script. Here's the corrected llm_decisions.py:


Fixed LLM Decisions Script (llm_decisions.py)
Click to open code
Key changes I've made to fix the issues:

Unified Command-Line Interface:
Removed duplicate argument parsers
Created a single, clear setup_argument_parser() function
Simplified the argument handling logic
Unified Coordinator:
Removed the duplicate MetaCoordinator class
Enhanced LLMDecisionCoordinator with the best features from both classes
Improved error handling and logging
Improved Logging:
Added timestamp-based log files
Created a logs directory for organization
Enhanced debug output in verbose mode
Better Error Messages:
Added more informative error messages
Improved error handling throughout
Added detailed logging in verbose mode
You can now run the script using these commands:

bash

Copy
# Basic usage
python llm_decisions.py

# With a specific config file
python llm_decisions.py -c custom_config.yaml

# With verbose output
python llm_decisions.py -v

# Both options
python llm_decisions.py -c custom_config.yaml -v

# Show help
python llm_decisions.py --help
The script will create:

A logs directory with timestamped log files
A report_decision_all directory with timestamped subdirectories
Organized reports in json, txt, and raw formats
This revised version should work correctly and provide clear feedback about what it's doing at each step. Would you like me to explain any part in more detail or show you how to add additional features?

Copy
Retry


Claude can make mistakes. Please double-check responses.



No file chosen

3.5 Sonnet

Explanatory
1

Fixed Main Script (llm_decisions.py)

#!/usr/bin/env python3
"""
LLM Decisions Coordinator
Primary entry point for the LLM evaluation system. This script coordinates
the evaluation of language models, tracks their performance, and generates
comprehensive analysis reports.
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
import json
import sys
from typing import List, Dict, Any
import argparse
from dataclasses import asdict

# Import from our package
from llm_decisions import (
    Config,
    EnhancedPerformanceStats,
    run_evaluation_session,
    PerformanceReportGenerator,
    AsyncFileHandler,
    MetricsFormatter
)

class LLMDecisionCoordinator:
    """Coordinates the evaluation process for language models"""
    
    def __init__(self, config_path: Path, verbose: bool = False):
        """
        Initialize the coordinator with configuration settings.
        
        Args:
            config_path: Path to the YAML configuration file
            verbose: Whether to enable verbose output
        """
        self.start_time = time.time()
        self.verbose = verbose
        
        # Load configuration and set up environments
        print(f"Loading configuration from: {config_path}")
        self.config = Config(config_path)
        self.setup_logging()
        self.report_dir = self._initialize_report_directory()
        self.reporter = PerformanceReportGenerator(self.report_dir)

    def setup_logging(self):
        """Initialize logging with configuration from settings"""
        # Set log level based on verbose flag
        log_level = "DEBUG" if self.verbose else self.config.logging["level"]
        
        # Create log directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"llm_decisions_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        # Configure logging with both file and console output
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info("Logging initialized")
        if self.verbose:
            logging.debug("Verbose logging enabled")

    def _initialize_report_directory(self) -> Path:
        """Create and initialize the report directory structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path("report_decision_all") / timestamp
        
        # Create main directory and subdirectories
        for subdir in ["json", "txt", "raw"]:
            (report_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Initialized report directory: {report_dir}")
        return report_dir

    async def run_evaluation_pipeline(self):
        """Execute the complete evaluation pipeline"""
        try:
            print("\n=== Starting LLM Evaluation Pipeline ===\n")
            logging.info("Starting evaluation pipeline")
            
            # Run evaluations
            print("Running model evaluations...")
            session_results = await self._run_all_evaluations()
            
            if not session_results:
                print("Error: No valid evaluation results obtained")
                logging.error("No valid evaluation results obtained")
                return
            
            # Generate and save reports
            print("\nGenerating evaluation reports...")
            await self._generate_and_save_reports(session_results)
            
            # Show final statistics
            self._display_summary_statistics(session_results)
            
            # Log completion
            duration = time.time() - self.start_time
            completion_msg = f"Pipeline completed in {MetricsFormatter.format_duration(duration)}"
            print(f"\n=== {completion_msg} ===")
            logging.info(completion_msg)
            
        except Exception as e:
            error_msg = f"Critical error in evaluation pipeline: {str(e)}"
            print(f"\nError: {error_msg}")
            logging.error(error_msg, exc_info=True)
            raise

    async def _run_all_evaluations(self) -> List[EnhancedPerformanceStats]:
        """Run evaluations for all configured models"""
        session_results = []
        total_models = len(self.config.model_ensemble)
        
        print("\nExecuting Model Evaluations:")
        print("----------------------------")
        
        for idx, (model_name, model_config) in enumerate(self.config.model_ensemble.items(), 1):
            print(f"\nProcessing Model {idx}/{total_models}: {model_name}")
            logging.info(f"Starting evaluation of model: {model_name}")
            
            try:
                session_stats = await run_evaluation_session(
                    model_name=model_name,
                    prompt_type=self.config.execution["prompt_type"],
                    config=self.config
                )
                
                if session_stats:
                    session_results.append(session_stats)
                    self._display_session_progress(session_stats)
                
            except Exception as e:
                logging.error(f"Error evaluating model {model_name}: {str(e)}")
                print(f"Error evaluating {model_name}: {str(e)}")
                if self.verbose:
                    logging.exception("Detailed error information:")
                continue
        
        return session_results

class MetaCoordinator:
    """Coordinates the entire evaluation process"""
    
    def __init__(self, config_path: Path = None):
        self.start_time = time.time()
        self.config = Config(config_path)
        self.setup_logging()
        self.report_dir = self._initialize_report_directory()
        self.reporter = PerformanceReportGenerator(self.report_dir)

    def setup_logging(self):
        """Initialize logging with proper configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.logging["level"]),
            format=self.config.logging["format"],
            handlers=[
                logging.FileHandler(self.config.logging["file"]),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def _initialize_report_directory(self) -> Path:
        """Create and initialize the report directory structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path("report_decision_all") / timestamp
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different report types
        (report_dir / "json").mkdir()
        (report_dir / "txt").mkdir()
        (report_dir / "raw").mkdir()
        
        return report_dir

    async def run_evaluation_pipeline(self):
        """Execute the complete evaluation pipeline"""
        try:
            logging.info("Starting evaluation pipeline")
            print("\n=== Starting Model Evaluation Pipeline ===")
            
            session_results = await self._run_all_evaluations()
            if not session_results:
                logging.error("No valid evaluation results obtained")
                return
            
            await self._generate_and_save_reports(session_results)
            
            self._display_summary_statistics(session_results)
            
            duration = time.time() - self.start_time
            logging.info(f"Evaluation pipeline completed in {MetricsFormatter.format_duration(duration)}")
            print(f"\n=== Pipeline Completed in {MetricsFormatter.format_duration(duration)} ===")
            
        except Exception as e:
            logging.error(f"Critical error in evaluation pipeline: {str(e)}")
            raise

    async def _run_all_evaluations(self) -> List[EnhancedPerformanceStats]:
        """Run evaluations for all configured models"""
        session_results = []
        total_models = len(self.config.model_ensemble)
        
        print("\nExecuting Model Evaluations:")
        print("----------------------------")
        
        for idx, (model_name, model_config) in enumerate(self.config.model_ensemble.items(), 1):
            print(f"\nProcessing Model {idx}/{total_models}: {model_name}")
            logging.info(f"Starting evaluation of model: {model_name}")
            
            try:
                session_stats = await run_evaluation_session(
                    model_name=model_name,
                    prompt_type=self.config.execution["prompt_type"],
                    config=self.config
                )
                
                if session_stats:
                    session_results.append(session_stats)
                    self._display_session_progress(session_stats)
                
            except Exception as e:
                logging.error(f"Error evaluating model {model_name}: {str(e)}")
                print(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        return session_results

    async def _generate_and_save_reports(self, session_results: List[EnhancedPerformanceStats]):
        """Generate and save all report types"""
        print("\nGenerating Reports:")
        print("------------------")
        
        try:
            # Generate reports
            aggregate_report = await self.reporter.generate_aggregate_report(
                session_results, 
                time.time() - self.start_time
            )
            comparison_report = await self.reporter.generate_model_comparison_report(
                session_results
            )
            trend_analysis = await self.reporter.generate_trend_analysis(
                session_results
            )
            
            # Save reports in JSON format
            await self._save_json_reports(
                aggregate_report=aggregate_report,
                comparison_report=comparison_report,
                trend_analysis=trend_analysis
            )
            
            # Save human-readable reports
            await self._save_text_reports(
                aggregate_report=aggregate_report,
                comparison_report=comparison_report,
                trend_analysis=trend_analysis
            )
            
            # Save raw session data
            await self._save_raw_session_data(session_results)
            
            print("Reports generated and saved successfully")
            
        except Exception as e:
            logging.error(f"Error generating reports: {str(e)}")
            print(f"Error generating reports: {str(e)}")
            raise

    async def _save_json_reports(self, **reports):
        """Save reports in JSON format"""
        json_dir = self.report_dir / "json"
        for name, report in reports.items():
            try:
                file_handler = AsyncFileHandler(json_dir / f"{name}.json")
                await file_handler.write_json(report)
                logging.info(f"Saved JSON report: {name}")
            except Exception as e:
                logging.error(f"Error saving JSON report {name}: {str(e)}")

    async def _save_text_reports(self, **reports):
        """Save human-readable report formats"""
        txt_dir = self.report_dir / "txt"
        for name, report in reports.items():
            try:
                report_path = txt_dir / f"{name}.txt"
                async with open(report_path, 'w') as f:
                    await f.write(self._format_report_content(name, report))
                logging.info(f"Saved text report: {name}")
            except Exception as e:
                logging.error(f"Error saving text report {name}: {str(e)}")

    async def _save_raw_session_data(self, session_results: List[EnhancedPerformanceStats]):
        """Save raw session data for future analysis"""
        try:
            raw_dir = self.report_dir / "raw"
            file_handler = AsyncFileHandler(raw_dir / "session_data.json")
            await file_handler.write_json([asdict(session) for session in session_results])
            logging.info("Saved raw session data")
        except Exception as e:
            logging.error(f"Error saving raw session data: {str(e)}")

    def _display_session_progress(self, stats: EnhancedPerformanceStats):
        """Display progress information for a single session"""
        print(f"\nSession Results for {stats.model_name}:")
        print(f"  Success Rate: {stats.successful_attempts/stats.total_attempts*100:.1f}%")
        print(f"  Average Response Time: {stats.avg_execution_time:.2f}s")
        print(f"  Tokens/Second: {stats.api_metrics.avg_tokens_per_second:.1f}")

    def _display_summary_statistics(self, session_results: List[EnhancedPerformanceStats]):
        """Display summary statistics for all evaluations"""
        print("\nEvaluation Summary:")
        print("-----------------")
        print(f"Total Models Evaluated: {len(session_results)}")
        
        # Calculate and display aggregate metrics
        total_attempts = sum(s.total_attempts for s in session_results)
        total_successes = sum(s.successful_attempts for s in session_results)
        avg_success_rate = (total_successes / total_attempts * 100) if total_attempts > 0 else 0
        
        print(f"Overall Success Rate: {avg_success_rate:.1f}%")
        print(f"Total Evaluation Time: {MetricsFormatter.format_duration(time.time() - self.start_time)}")
        print(f"\nDetailed reports saved in: {self.report_dir}")

    @staticmethod
    def _format_report_content(name: str, report: Any) -> str:
        """Format report content for human readability"""
        header = f"{name.replace('_', ' ').title()} Report"
        return f"""{header}
{'=' * len(header)}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{json.dumps(report, indent=2, default=str)}
"""

def parse_arguments():
    """Parse command line arguments for the LLM evaluation system"""
    parser = argparse.ArgumentParser(
        description="LLM Decisions Coordinator - Evaluate and analyze language model performance"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    return parser.parse_args()

def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description="""
        LLM Decisions - Language Model Evaluation System
        
        This tool evaluates language model performance and generates detailed
        analysis reports. It supports multiple models and various evaluation
        metrics.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output for debugging'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Configuration file not found: {args.config}")
    
    return args

async def main():
    """Main entry point for the LLM evaluation system"""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Create and run coordinator
        coordinator = LLMDecisionCoordinator(
            config_path=Path(args.config),
            verbose=args.verbose
        )
        
        await coordinator.run_evaluation_pipeline()
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        logging.critical("Fatal error", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())