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

    # [Previous implementation of other methods remains the same...]

def setup_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser"""
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
    
    return parser

async def main():
    """Main entry point for the LLM evaluation system"""
    # Set up argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Create and run coordinator
        coordinator = LLMDecisionCoordinator(
            config_path=config_path,
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