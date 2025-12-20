"""Model versioning and promotion system.

This module provides tools for managing model versions, promoting models
to production, and ensuring safe model deployments.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    
    name: str
    version: str
    path: str
    created_at: str
    metrics: dict
    status: str = "candidate"  # candidate, production, archived
    promoted_at: Optional[str] = None
    promoted_by: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ModelMetadata:
        """Create from dictionary."""
        return cls(**data)


class ModelRegistry:
    """Registry for managing model versions and promotions."""

    def __init__(self, models_dir: str | Path):
        """Initialize the model registry.

        Args:
            models_dir: Directory where models are stored
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.models_dir / "registry.json"
        self._load_registry()

    def _load_registry(self) -> None:
        """Load the model registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                data = json.load(f)
                self.models = [ModelMetadata.from_dict(m) for m in data.get("models", [])]
        else:
            self.models = []
            self._save_registry()

    def _save_registry(self) -> None:
        """Save the model registry to disk."""
        data = {
            "models": [m.to_dict() for m in self.models],
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.registry_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved model registry with {len(self.models)} models")

    def register_model(
        self,
        name: str,
        version: str,
        path: str,
        metrics: dict,
        notes: Optional[str] = None,
    ) -> ModelMetadata:
        """Register a new model in the registry.

        Args:
            name: Model name (e.g., "xgboost_v1")
            version: Version string (e.g., "1.0.0" or timestamp)
            path: Relative path to model file
            metrics: Dictionary of model metrics (accuracy, precision, etc.)
            notes: Optional notes about the model

        Returns:
            ModelMetadata for the registered model
        """
        metadata = ModelMetadata(
            name=name,
            version=version,
            path=path,
            created_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            status="candidate",
            notes=notes,
        )
        
        self.models.append(metadata)
        self._save_registry()
        
        logger.info(f"Registered model {name} version {version}")
        return metadata

    def promote_to_production(
        self,
        name: str,
        version: str,
        promoted_by: str = "system",
    ) -> Optional[ModelMetadata]:
        """Promote a model to production status.

        This will:
        1. Set the specified model status to "production"
        2. Demote any existing production models to "archived"
        3. Create a production.json pointer to the new production model

        Args:
            name: Model name
            version: Model version
            promoted_by: Name of person/system promoting the model

        Returns:
            ModelMetadata for the promoted model, or None if not found
        """
        # Find the model to promote
        target_model = None
        for model in self.models:
            if model.name == name and model.version == version:
                target_model = model
                break

        if not target_model:
            logger.error(f"Model {name} version {version} not found in registry")
            return None

        # Demote existing production models
        for model in self.models:
            if model.status == "production":
                model.status = "archived"
                logger.info(f"Archived previous production model: {model.name} v{model.version}")

        # Promote the target model
        target_model.status = "production"
        target_model.promoted_at = datetime.now(timezone.utc).isoformat()
        target_model.promoted_by = promoted_by

        # Save registry
        self._save_registry()

        # Create production pointer
        production_pointer = {
            "name": target_model.name,
            "version": target_model.version,
            "path": target_model.path,
            "promoted_at": target_model.promoted_at,
            "promoted_by": target_model.promoted_by,
            "metrics": target_model.metrics,
        }
        
        production_file = self.models_dir / "production.json"
        with open(production_file, "w") as f:
            json.dump(production_pointer, f, indent=2)

        logger.info(f"Promoted {name} v{version} to production")
        return target_model

    def get_production_model(self) -> Optional[ModelMetadata]:
        """Get the current production model.

        Returns:
            ModelMetadata for the production model, or None if no model is in production
        """
        for model in self.models:
            if model.status == "production":
                return model
        return None

    def list_models(
        self,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ModelMetadata]:
        """List models in the registry.

        Args:
            status: Optional status filter (candidate, production, archived)
            limit: Optional limit on number of results

        Returns:
            List of ModelMetadata objects
        """
        models = self.models
        
        if status:
            models = [m for m in models if m.status == status]
        
        # Sort by created_at descending
        models = sorted(models, key=lambda m: m.created_at, reverse=True)
        
        if limit:
            models = models[:limit]
        
        return models

    def compare_models(
        self,
        name1: str,
        version1: str,
        name2: str,
        version2: str,
    ) -> dict:
        """Compare metrics between two models.

        Args:
            name1: First model name
            version1: First model version
            name2: Second model name
            version2: Second model version

        Returns:
            Dictionary with comparison results
        """
        model1 = None
        model2 = None
        
        for model in self.models:
            if model.name == name1 and model.version == version1:
                model1 = model
            if model.name == name2 and model.version == version2:
                model2 = model
        
        if not model1 or not model2:
            return {"error": "One or both models not found"}
        
        comparison = {
            "model1": {
                "name": model1.name,
                "version": model1.version,
                "metrics": model1.metrics,
                "status": model1.status,
            },
            "model2": {
                "name": model2.name,
                "version": model2.version,
                "metrics": model2.metrics,
                "status": model2.status,
            },
            "differences": {},
        }
        
        # Calculate metric differences
        for metric in model1.metrics:
            if metric in model2.metrics:
                diff = model1.metrics[metric] - model2.metrics[metric]
                comparison["differences"][metric] = {
                    "model1": model1.metrics[metric],
                    "model2": model2.metrics[metric],
                    "difference": diff,
                    "percentage_change": (diff / model2.metrics[metric] * 100) if model2.metrics[metric] != 0 else None,
                }
        
        return comparison

