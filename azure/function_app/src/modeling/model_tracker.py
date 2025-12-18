"""
Model performance tracking and versioning system.

Tracks model performance over time and manages model versions.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelVersion:
    """Model version metadata."""
    version: str
    model_type: str  # "spreads", "totals", "moneyline", "ensemble"
    algorithm: str  # "logistic", "gradient_boosting", "ensemble"
    trained_at: str
    train_samples: int
    test_samples: int
    features_count: int
    feature_names: List[str]
    metrics: Dict[str, float]
    file_path: str
    notes: Optional[str] = None


@dataclass
class ModelPerformance:
    """Model performance metrics over time."""
    model_id: str
    date: str
    predictions_count: int
    accuracy: float
    log_loss: float
    roi: float
    cover_rate: float
    notes: Optional[str] = None


class ModelTracker:
    """
    Track model versions and performance.

    Stores metadata in data/processed/models/manifest.json
    """

    def __init__(self, manifest_path: Optional[str] = None):
        from src.config import settings

        if manifest_path is None:
            manifest_path = os.path.join(
                settings.data_processed_dir, "models", "manifest.json"
            )

        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        """Load manifest from disk."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)

        return {
            'versions': [],
            'performance': [],
            'active_models': {},  # model_type -> version
        }

    def _save_manifest(self):
        """Save manifest to disk."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def register_version(
        self,
        version: ModelVersion,
        set_active: bool = True
    ) -> None:
        """
        Register a new model version.

        Args:
            version: ModelVersion object
            set_active: Whether to set this as the active version for this model type
        """
        # Add to versions list
        self.manifest['versions'].append(asdict(version))

        # Optionally set as active
        if set_active:
            self.manifest['active_models'][version.model_type] = version.version

        self._save_manifest()
        print(f"[OK] Registered model version: {version.version} ({version.model_type})")

    def log_performance(self, performance: ModelPerformance) -> None:
        """
        Log model performance metrics.

        Args:
            performance: ModelPerformance object
        """
        self.manifest['performance'].append(asdict(performance))
        self._save_manifest()

    def get_active_version(self, model_type: str) -> Optional[str]:
        """Get currently active version for a model type."""
        active_models = self.manifest.get('active_models', {})
        return active_models.get(model_type)

    def get_version_info(self, version: str) -> Optional[Dict]:
        """Get metadata for a specific version."""
        for v in self.manifest['versions']:
            if v['version'] == version:
                return v
        return None

    def get_performance_history(
        self,
        model_id: str,
        days: Optional[int] = None
    ) -> List[Dict]:
        """Get performance history for a model."""
        history = [p for p in self.manifest['performance'] if p['model_id'] == model_id]

        if days:
            # Filter to recent days
            cutoff = datetime.now().timestamp() - (days * 86400)
            history = [
                p for p in history
                if datetime.fromisoformat(p['date']).timestamp() > cutoff
            ]

        return sorted(history, key=lambda x: x['date'])

    def compare_versions(
        self,
        version1: str,
        version2: str
    ) -> Dict[str, float]:
        """
        Compare two model versions.

        Returns:
            Dict of metric differences
        """
        v1 = self.get_version_info(version1)
        v2 = self.get_version_info(version2)

        if not v1 or not v2:
            raise ValueError("Version not found")

        metrics1 = v1['metrics']
        metrics2 = v2['metrics']

        comparison = {}
        for key in metrics1.keys():
            if key in metrics2:
                comparison[key] = metrics2[key] - metrics1[key]

        return comparison

    def list_versions(self, model_type: Optional[str] = None) -> List[Dict]:
        """List all versions, optionally filtered by model type."""
        versions = self.manifest['versions']

        if model_type:
            versions = [v for v in versions if v['model_type'] == model_type]

        return sorted(versions, key=lambda x: x['trained_at'], reverse=True)

    def get_best_version(self, model_type: str, metric: str = 'accuracy') -> Optional[str]:
        """
        Get best performing version for a model type.

        Args:
            model_type: Type of model
            metric: Metric to optimize (accuracy, log_loss, roi, etc.)

        Returns:
            Version string of best model
        """
        versions = self.list_versions(model_type)

        if not versions:
            return None

        # For log_loss, lower is better
        if 'loss' in metric.lower():
            best = min(versions, key=lambda x: x['metrics'].get(metric, float('inf')))
        else:
            best = max(versions, key=lambda x: x['metrics'].get(metric, 0))

        return best['version']

    def promote_to_production(self, version: str) -> None:
        """
        Promote a version to production.

        Creates a production.json file pointing to the specified version.
        """
        version_info = self.get_version_info(version)
        if not version_info:
            raise ValueError(f"Version {version} not found")

        production_path = self.manifest_path.parent / 'production.json'
        with open(production_path, 'w') as f:
            json.dump({
                'version': version,
                'model_type': version_info['model_type'],
                'promoted_at': datetime.now().isoformat(),
                'file_path': version_info['file_path']
            }, f, indent=2)

        print(f"[OK] Promoted {version} to production")

    def get_production_version(self) -> Optional[Dict]:
        """Get currently deployed production version."""
        production_path = self.manifest_path.parent / 'production.json'

        if not production_path.exists():
            return None

        with open(production_path, 'r') as f:
            return json.load(f)

    def generate_report(self) -> str:
        """Generate a performance report."""
        lines = []
        lines.append("="*80)
        lines.append("MODEL PERFORMANCE REPORT")
        lines.append("="*80)
        lines.append("")

        # Active models
        lines.append("ACTIVE MODELS:")
        for model_type, version in self.manifest['active_models'].items():
            version_info = self.get_version_info(version)
            if version_info:
                metrics = version_info['metrics']
                lines.append(f"  {model_type:15} v{version:10} - Acc: {metrics.get('accuracy', 0):.1%}, LogLoss: {metrics.get('log_loss', 0):.4f}")

        lines.append("")

        # Recent performance
        lines.append("RECENT PERFORMANCE (Last 7 days):")
        for model_type in self.manifest['active_models'].keys():
            version = self.manifest['active_models'][model_type]
            history = self.get_performance_history(version, days=7)

            if history:
                avg_acc = sum(p['accuracy'] for p in history) / len(history)
                avg_roi = sum(p['roi'] for p in history) / len(history)
                lines.append(f"  {model_type:15} - Predictions: {len(history)}, Avg Acc: {avg_acc:.1%}, Avg ROI: {avg_roi:.1%}")
            else:
                lines.append(f"  {model_type:15} - No recent predictions")

        lines.append("")
        lines.append("="*80)

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    tracker = ModelTracker()
    print(tracker.generate_report())
