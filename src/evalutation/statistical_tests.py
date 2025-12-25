"""
Statistical significance testing for model comparison.

Provides functionality to:
- Perform paired t-tests for model comparison
- Calculate confidence intervals
- Generate comparison reports
"""
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple
from src.utils.metrics import RecEvaluator


class StatisticalComparator:
    """
    Performs statistical significance testing between two models.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical comparator.
        
        Args:
            confidence_level: Confidence level for intervals (default: 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def compare_models(
        self,
        predictions1: List[Dict],
        predictions2: List[Dict],
        metrics_names: List[str] = None
    ) -> Dict[str, Dict]:
        """
        Compare two models using paired t-test.
        
        Args:
            predictions1: Predictions from first model
            predictions2: Predictions from second model
            metrics_names: List of metric names to compare (default: all)
            
        Returns:
            Dictionary with comparison results for each metric
        """
        if metrics_names is None:
            metrics_names = ["ndcg_at_10", "ndcg_at_5", "auc", "mrr"]
        
        # Extract per-sample metrics
        samples1 = self._extract_sample_metrics(predictions1)
        samples2 = self._extract_sample_metrics(predictions2)
        
        # Verify same number of samples
        if len(predictions1) != len(predictions2):
            raise ValueError(
                f"Number of predictions must match: {len(predictions1)} vs {len(predictions2)}"
            )
        
        # Perform tests for each metric
        comparison_results = {}
        for metric_name in metrics_names:
            if metric_name not in samples1 or metric_name not in samples2:
                continue
            
            values1 = samples1[metric_name]
            values2 = samples2[metric_name]
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(values1, values2)
            
            # Calculate difference statistics
            diff = np.array(values1) - np.array(values2)
            mean_diff = np.mean(diff)
            std_diff = np.std(diff, ddof=1)
            n = len(diff)
            
            # Confidence interval for the difference
            ci = self._calculate_confidence_interval(diff)
            
            # Effect size (Cohen's d)
            cohens_d = mean_diff / std_diff if std_diff > 0 else 0
            
            # Determine which model is better
            better_model = "model1" if mean_diff > 0 else "model2" if mean_diff < 0 else "tie"
            
            comparison_results[metric_name] = {
                "model1_mean": np.mean(values1),
                "model1_std": np.std(values1),
                "model2_mean": np.mean(values2),
                "model2_std": np.std(values2),
                "difference_mean": mean_diff,
                "difference_std": std_diff,
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant_at_0.05": p_value < 0.05,
                "significant_at_0.01": p_value < 0.01,
                "confidence_interval": ci,
                "confidence_level": self.confidence_level,
                "cohens_d": cohens_d,
                "effect_size": self._interpret_effect_size(cohens_d),
                "better_model": better_model,
                "num_samples": n
            }
        
        return comparison_results
    
    def _extract_sample_metrics(self, predictions: List[Dict]) -> Dict[str, List[float]]:
        """
        Extract per-sample metrics from predictions.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Dictionary mapping metric names to lists of values
        """
        sample_metrics = {
            "ndcg_at_10": [],
            "ndcg_at_5": [],
            "auc": [],
            "mrr": []
        }
        
        for pred in predictions:
            y_true = np.array(pred["labels"])
            y_score = np.array(pred["scores"])
            
            # Compute metrics
            metrics = RecEvaluator.evaluate_all(y_true, y_score)
            
            sample_metrics["ndcg_at_10"].append(metrics.ndcg_at_10)
            sample_metrics["ndcg_at_5"].append(metrics.ndcg_at_5)
            sample_metrics["auc"].append(metrics.auc)
            sample_metrics["mrr"].append(metrics.mrr)
        
        return sample_metrics
    
    def _calculate_confidence_interval(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Calculate confidence interval for data.
        
        Args:
            data: Array of values
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        
        # t-distribution critical value
        t_crit = stats.t.ppf((1 + self.confidence_level) / 2, n - 1)
        
        # Margin of error
        margin = t_crit * std / np.sqrt(n)
        
        return (mean - margin, mean + margin)
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """
        Interpret Cohen's d effect size.
        
        Args:
            cohens_d: Cohen's d value
            
        Returns:
            Interpretation string
        """
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def generate_comparison_report(
        self,
        comparison_results: Dict[str, Dict],
        model1_name: str = "Model 1",
        model2_name: str = "Model 2"
    ) -> str:
        """
        Generate a formatted comparison report.
        
        Args:
            comparison_results: Results from compare_models
            model1_name: Name of first model
            model2_name: Name of second model
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "="*80,
            f"STATISTICAL COMPARISON: {model1_name} vs {model2_name}",
            "="*80,
            ""
        ]
        
        # Summary table
        report_lines.append(f"{'Metric':<15} {model1_name:<12} {model2_name:<12} {'Diff':<12} {'p-value':<10} {'Sig.':<6} {'Effect':<10}")
        report_lines.append("-"*80)
        
        for metric_name, results in comparison_results.items():
            sig_marker = "***" if results["significant_at_0.01"] else ("**" if results["significant_at_0.05"] else "")
            
            report_lines.append(
                f"{metric_name:<15} "
                f"{results['model1_mean']:>6.4f}±{results['model1_std']:>4.3f} "
                f"{results['model2_mean']:>6.4f}±{results['model2_std']:>4.3f} "
                f"{results['difference_mean']:>+6.4f}±{results['difference_std']:>4.3f} "
                f"{results['p_value']:>10.4f} "
                f"{sig_marker:<6} "
                f"{results['effect_size']:<10}"
            )
        
        report_lines.append("")
        report_lines.append("Significance levels: ** p<0.05, *** p<0.01")
        report_lines.append("")
        
        # Detailed results
        report_lines.append("="*80)
        report_lines.append("DETAILED RESULTS")
        report_lines.append("="*80)
        
        for metric_name, results in comparison_results.items():
            report_lines.append(f"\n{metric_name.upper()}:")
            report_lines.append(f"  {model1_name}: {results['model1_mean']:.4f} ± {results['model1_std']:.4f}")
            report_lines.append(f"  {model2_name}: {results['model2_mean']:.4f} ± {results['model2_std']:.4f}")
            report_lines.append(f"  Difference: {results['difference_mean']:+.4f} ± {results['difference_std']:.4f}")
            report_lines.append(f"  {self.confidence_level*100:.0f}% CI: [{results['confidence_interval'][0]:+.4f}, {results['confidence_interval'][1]:+.4f}]")
            report_lines.append(f"  t-statistic: {results['t_statistic']:.4f}")
            report_lines.append(f"  p-value: {results['p_value']:.4f}")
            report_lines.append(f"  Cohen's d: {results['cohens_d']:.4f} ({results['effect_size']})")
            
            if results['significant_at_0.05']:
                report_lines.append(f"  ✓ Statistically significant at α=0.05")
                report_lines.append(f"  Better model: {results['better_model']}")
            else:
                report_lines.append(f"  ✗ Not statistically significant at α=0.05")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        # Overall conclusion
        report_lines.append("\nOVERALL CONCLUSION:")
        
        # Count significant improvements
        sig_improvements_model1 = sum(
            1 for r in comparison_results.values()
            if r['significant_at_0.05'] and r['better_model'] == 'model1'
        )
        sig_improvements_model2 = sum(
            1 for r in comparison_results.values()
            if r['significant_at_0.05'] and r['better_model'] == 'model2'
        )
        
        if sig_improvements_model1 > sig_improvements_model2:
            report_lines.append(f"  {model1_name} shows statistically significant improvements in {sig_improvements_model1} metric(s).")
        elif sig_improvements_model2 > sig_improvements_model1:
            report_lines.append(f"  {model2_name} shows statistically significant improvements in {sig_improvements_model2} metric(s).")
        else:
            report_lines.append(f"  No clear winner. Both models show similar performance.")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        return "\n".join(report_lines)


def perform_bootstrap_test(
    predictions1: List[Dict],
    predictions2: List[Dict],
    metric_name: str = "ndcg_at_10",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict:
    """
    Perform bootstrap test for model comparison.
    
    Args:
        predictions1: Predictions from first model
        predictions2: Predictions from second model
        metric_name: Metric to compare
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        
    Returns:
        Dictionary with bootstrap test results
    """
    # Extract metric values
    def extract_metric(predictions, metric_name):
        values = []
        for pred in predictions:
            y_true = np.array(pred["labels"])
            y_score = np.array(pred["scores"])
            metrics = RecEvaluator.evaluate_all(y_true, y_score)
            values.append(getattr(metrics, metric_name))
        return np.array(values)
    
    values1 = extract_metric(predictions1, metric_name)
    values2 = extract_metric(predictions2, metric_name)
    
    # Observed difference
    observed_diff = np.mean(values1) - np.mean(values2)
    
    # Bootstrap sampling
    n_samples = len(values1)
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_values1 = values1[indices]
        boot_values2 = values2[indices]
        
        # Calculate difference
        boot_diff = np.mean(boot_values1) - np.mean(boot_values2)
        bootstrap_diffs.append(boot_diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_diffs, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
    
    # p-value (proportion of bootstrap samples with opposite sign)
    p_value = np.mean(np.sign(bootstrap_diffs) != np.sign(observed_diff))
    
    return {
        "metric": metric_name,
        "observed_difference": observed_diff,
        "bootstrap_mean": np.mean(bootstrap_diffs),
        "bootstrap_std": np.std(bootstrap_diffs),
        "confidence_interval": (ci_lower, ci_upper),
        "confidence_level": confidence_level,
        "p_value": p_value,
        "n_bootstrap": n_bootstrap,
        "significant": ci_lower > 0 or ci_upper < 0
    }
