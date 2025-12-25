"""
Submission file formatter for MIND leaderboard.

This module provides utilities to format model predictions into the
official MIND dataset submission format for leaderboard evaluation.

MIND Submission Format:
    impression_id [rank1,rank2,rank3,...]
    
    Where ranks are 1-indexed positions of candidate news items
    sorted by predicted relevance scores (highest to lowest).
"""
import json
import zipfile
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime
import numpy as np


class SubmissionFormatter:
    """
    Formatter for MIND leaderboard submissions.
    
    Handles conversion of model predictions to the official MIND format,
    validation, and metadata generation.
    """
    
    def __init__(self, metadata: Optional[Dict] = None):
        """
        Initialize submission formatter.
        
        Args:
            metadata: Optional metadata dictionary to include in submission
        """
        self.metadata = metadata or {}
        self.metadata.setdefault("created_at", datetime.now().isoformat())
        self.metadata.setdefault("format_version", "1.0")
    
    def format_predictions(
        self,
        scores: Union[List[np.ndarray], np.ndarray],
        impression_ids: List[str]
    ) -> List[List[int]]:
        """
        Convert prediction scores to ranked indices.
        
        Args:
            scores: Prediction scores for each impression
                   Can be list of arrays or 2D array (num_impressions, num_candidates)
            impression_ids: List of impression IDs
            
        Returns:
            List of ranked indices (1-indexed) for each impression
        """
        if isinstance(scores, np.ndarray) and len(scores.shape) == 2:
            # Convert 2D array to list of arrays
            scores = [scores[i] for i in range(len(scores))]
        
        if len(scores) != len(impression_ids):
            raise ValueError(
                f"Number of score arrays ({len(scores)}) must match "
                f"number of impression IDs ({len(impression_ids)})"
            )
        
        ranked_predictions = []
        for score_array in scores:
            # Sort indices by score (descending order)
            # argsort gives ascending order, so we reverse it
            ranked_indices = np.argsort(score_array)[::-1]
            # Convert to 1-indexed ranks
            ranks = (ranked_indices + 1).tolist()
            ranked_predictions.append(ranks)
        
        return ranked_predictions
    
    def validate_submission(
        self,
        predictions: List[List[int]],
        impression_ids: List[str]
    ) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate submission format.
        
        Args:
            predictions: List of ranked indices for each impression
            impression_ids: List of impression IDs
            
        Returns:
            Dictionary with validation results:
                - valid: bool indicating if submission is valid
                - errors: list of error messages (empty if valid)
        """
        errors = []
        
        # Check lengths match
        if len(predictions) != len(impression_ids):
            errors.append(
                f"Mismatch: {len(predictions)} predictions vs "
                f"{len(impression_ids)} impression IDs"
            )
        
        # Check impression IDs are unique
        if len(impression_ids) != len(set(impression_ids)):
            errors.append("Duplicate impression IDs found")
        
        # Check each prediction
        for i, (imp_id, ranks) in enumerate(zip(impression_ids, predictions)):
            # Check impression ID is not empty
            if not imp_id or not str(imp_id).strip():
                errors.append(f"Empty impression ID at index {i}")
            
            # Check ranks is not empty
            if not ranks:
                errors.append(f"Empty ranks for impression {imp_id}")
                continue
            
            # Check all ranks are positive integers
            if not all(isinstance(r, int) and r > 0 for r in ranks):
                errors.append(
                    f"Invalid ranks for impression {imp_id}: "
                    f"all ranks must be positive integers"
                )
            
            # Check ranks are unique
            if len(ranks) != len(set(ranks)):
                errors.append(f"Duplicate ranks for impression {imp_id}")
            
            # Check ranks form a valid sequence (1 to N)
            expected_ranks = set(range(1, len(ranks) + 1))
            actual_ranks = set(ranks)
            if actual_ranks != expected_ranks:
                errors.append(
                    f"Invalid rank sequence for impression {imp_id}: "
                    f"expected {sorted(expected_ranks)}, got {sorted(actual_ranks)}"
                )
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def write_submission_file(
        self,
        predictions: List[List[int]],
        impression_ids: List[str],
        output_path: Union[str, Path],
        validate: bool = True
    ) -> Dict[str, Union[bool, int, List[str]]]:
        """
        Write predictions to submission file in MIND format.
        
        Format: impression_id [rank1,rank2,rank3,...]
        
        Args:
            predictions: List of ranked indices (1-indexed) for each impression
            impression_ids: List of impression IDs
            output_path: Path to output file
            validate: Whether to validate before writing
            
        Returns:
            Dictionary with write results:
                - success: bool indicating if write was successful
                - num_impressions: number of impressions written
                - errors: list of validation errors (if any)
        """
        output_path = Path(output_path)
        
        # Validate if requested
        if validate:
            validation_result = self.validate_submission(predictions, impression_ids)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "num_impressions": 0,
                    "errors": validation_result["errors"]
                }
        
        # Write submission file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for imp_id, ranks in zip(impression_ids, predictions):
                ranks_str = ",".join(map(str, ranks))
                f.write(f"{imp_id} [{ranks_str}]\n")
        
        return {
            "success": True,
            "num_impressions": len(impression_ids),
            "errors": []
        }
    
    def add_metadata(self, key: str, value: any):
        """Add metadata field."""
        self.metadata[key] = value
    
    def write_metadata_file(self, output_path: Union[str, Path]):
        """
        Write metadata to JSON file.
        
        Args:
            output_path: Path to output metadata file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
    
    def create_submission_package(
        self,
        predictions: List[List[int]],
        impression_ids: List[str],
        output_dir: Union[str, Path],
        package_name: str = "submission",
        include_metadata: bool = True
    ) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Create a complete submission package with predictions and metadata.
        
        Args:
            predictions: List of ranked indices for each impression
            impression_ids: List of impression IDs
            output_dir: Directory to save submission package
            package_name: Name for submission files (without extension)
            include_metadata: Whether to include metadata file
            
        Returns:
            Dictionary with package creation results:
                - success: bool indicating if package was created
                - prediction_file: path to prediction file
                - metadata_file: path to metadata file (if created)
                - errors: list of errors (if any)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write prediction file
        prediction_file = output_dir / f"{package_name}.txt"
        write_result = self.write_submission_file(
            predictions=predictions,
            impression_ids=impression_ids,
            output_path=prediction_file,
            validate=True
        )
        
        if not write_result["success"]:
            return {
                "success": False,
                "prediction_file": None,
                "metadata_file": None,
                "errors": write_result["errors"]
            }
        
        result = {
            "success": True,
            "prediction_file": str(prediction_file),
            "metadata_file": None,
            "errors": []
        }
        
        # Write metadata file if requested
        if include_metadata:
            metadata_file = output_dir / f"{package_name}_metadata.json"
            self.write_metadata_file(metadata_file)
            result["metadata_file"] = str(metadata_file)
        
        return result


def write_submission(predictions, impression_ids, output_path):
    """
    Legacy function for backward compatibility.
    
    Write predictions to submission file in MIND format.
    
    Args:
        predictions: list of list (rank indices, 1-indexed)
        impression_ids: list of impression IDs
        output_path: path to output file
    """
    formatter = SubmissionFormatter()
    formatter.write_submission_file(
        predictions=predictions,
        impression_ids=impression_ids,
        output_path=output_path,
        validate=False  # Legacy behavior: no validation
    )
