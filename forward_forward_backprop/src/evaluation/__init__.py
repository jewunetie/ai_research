"""
Evaluation utilities for Forward-Forward with Backprop experiments.
"""

from .metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    evaluate_model,
    linear_probing_evaluation,
    layer_wise_goodness_analysis,
    extract_features
)

from .visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    visualize_tsne,
    plot_layer_goodness,
    compare_approaches_plot
)

__all__ = [
    # Metrics
    'compute_accuracy',
    'compute_confusion_matrix',
    'evaluate_model',
    'linear_probing_evaluation',
    'layer_wise_goodness_analysis',
    'extract_features',
    # Visualization
    'plot_training_curves',
    'plot_confusion_matrix',
    'visualize_tsne',
    'plot_layer_goodness',
    'compare_approaches_plot'
]
