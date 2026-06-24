import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Import module directly from file
def import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import ensemble algorithms from file
current_dir = os.path.dirname(os.path.abspath(__file__))
ensemble_file = os.path.join(current_dir, "Ensemble_RF+XGB_CV.py")
ensemble_module = import_from_file("ensemble_model", ensemble_file)
EnsembleClassifier = ensemble_module.EnsembleClassifier
RandomForestModel = ensemble_module.RandomForestModel
XGBoostClassifier = ensemble_module.XGBoostClassifier


def load_data(file_path, sheet_name):
    """
    Load Excel data and return feature names
    """
    # Read Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Get feature names - Use original feature columns (1-13)
    feature_names = df.columns[1:14].tolist()

    # Separate features and labels
    X = df.iloc[:, 1:14].values
    y = df.iloc[:, 0].values

    # Convert labels to numeric
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y, le.classes_, feature_names


def calculate_permutation_importance(ensemble_model, X, y, positive_class, feature_names, n_repeats=10):
    """
    Calculate permutation feature importance
    """
    logging.info(f"Calculating permutation feature importance for class {positive_class}...")

    y_binary = (y == positive_class).astype(int)

    rf_component = ensemble_model.rf_component
    xgb_component = ensemble_model.xgb_component

    rf_component.original_indices = np.arange(len(X))
    xgb_component.original_indices = np.arange(len(X))

    rf_features = X.copy()
    xgb_features = X.copy()
    original_indices = np.arange(len(X))

    rf_data = (rf_features, y, original_indices)
    xgb_data = (xgb_features, y, original_indices)

    rf_balanced_data, xgb_balanced_data = ensemble_model.apply_data_augmentation(rf_data, xgb_data, positive_class)

    rf_X_balanced, rf_y_binary, _ = rf_balanced_data
    xgb_X_balanced, xgb_y_binary, _ = xgb_balanced_data

    # Train base models - GA version requires additional parameters
    fold_idx = 0

    # Get class name (convert numeric to string class name)
    class_names = ensemble_model.all_classes
    class_name = class_names[positive_class] if positive_class < len(class_names) else str(positive_class)

    logging.info("Optimizing RF algorithms hyperparameters...")
    rf_best_params = rf_component._optimize_hyperparameters(
        rf_X_balanced, rf_y_binary, class_name, fold_idx
    )
    rf_model = rf_component._create_model(rf_best_params)
    rf_model.fit(rf_X_balanced, rf_y_binary)

    logging.info("Optimizing XGBoost algorithms hyperparameters...")
    xgb_best_params = xgb_component._optimize_hyperparameters(
        xgb_X_balanced, xgb_y_binary, class_name, fold_idx
    )
    xgb_model = xgb_component._create_model(xgb_best_params)
    xgb_model.fit(xgb_X_balanced, xgb_y_binary)

    # Optimize ensemble weight
    rf_y_pred_proba = rf_model.predict_proba(X)[:, 1]
    xgb_y_pred_proba = xgb_model.predict_proba(X)[:, 1]
    best_ensemble_weight = ensemble_model._optimize_ensemble_weight(rf_y_pred_proba, xgb_y_pred_proba, y_binary)

    logging.info(f"Best XGBoost weight: {best_ensemble_weight:.2f}")

    # Calculate baseline performance
    ensemble_y_pred_proba = (1 - best_ensemble_weight) * rf_y_pred_proba + best_ensemble_weight * xgb_y_pred_proba
    from sklearn.metrics import roc_auc_score
    baseline_score = roc_auc_score(y_binary, ensemble_y_pred_proba)

    # Store importance scores for each feature
    importance_scores = []

    # Evaluate each feature
    def process_feature(feature_idx):
        feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature_{feature_idx}"
        logging.info(f"Processing feature {feature_name} ({feature_idx + 1}/{X.shape[1]})...")

        feature_scores = []

        # Repeat multiple times for stable results
        for repeat in range(n_repeats):
            logging.debug(f"Repeat {repeat + 1}/{n_repeats}")

            # Copy original features
            X_permuted = X.copy()

            # Randomly shuffle values of current feature
            np.random.shuffle(X_permuted[:, feature_idx])

            # Process permuted data (same processing flow as original data)
            rf_features_permuted = X_permuted.copy()
            xgb_features_permuted = X_permuted.copy()

            # Get base algorithms predictions
            rf_y_pred_proba_permuted = rf_model.predict_proba(rf_features_permuted)[:, 1]
            xgb_y_pred_proba_permuted = xgb_model.predict_proba(xgb_features_permuted)[:, 1]

            # Get ensemble prediction
            ensemble_y_pred_proba_permuted = (
                                                     1 - best_ensemble_weight) * rf_y_pred_proba_permuted + best_ensemble_weight * xgb_y_pred_proba_permuted

            # Calculate permuted score
            try:
                permuted_score = roc_auc_score(y_binary, ensemble_y_pred_proba_permuted)

                # Calculate importance (baseline minus permuted)
                importance = baseline_score - permuted_score
                feature_scores.append(importance)
            except Exception as e:
                logging.warning(f"Error calculating ROC AUC for feature {feature_name}: {str(e)}")
                feature_scores.append(0)

        # Calculate average importance score
        avg_importance = np.mean(feature_scores)
        logging.info(f"Average importance for feature {feature_name}: {avg_importance:.4f}")
        return avg_importance

    # Parallel processing of feature evaluation
    feature_indices = range(X.shape[1])
    importance_scores = Parallel(n_jobs=-1)(
        delayed(process_feature)(idx) for idx in feature_indices
    )

    return np.array(importance_scores)


def plot_feature_importance(importance_scores_dict, feature_names, output_path):
    """
    Plot feature importance bar charts for each class and combined
    """
    # Create separate plots for each class
    for class_name, scores in importance_scores_dict.items():
        plt.figure(figsize=(12, 8))

        plt.rcParams['font.family'] = 'Arial'

        sorted_idx = np.argsort(scores)

        bars = plt.barh(range(len(sorted_idx)), scores[sorted_idx], color='skyblue')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', va='center', fontsize=10)

        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx], fontsize=12)

        plt.title(f'Feature Importance Ranking - {class_name} (Permutation Method)', fontsize=16, pad=20)
        plt.xlabel('Feature Importance Score', fontsize=14, labelpad=10)

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        plt.tight_layout()
        filename = f'feature_importance_permutation_{class_name}.png'
        plt.savefig(os.path.join(output_path, filename), dpi=600, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved feature importance chart for {class_name}")


def save_importance_scores(importance_scores_dict, feature_names, output_path):
    """
    Save feature importance scores for all classes to Excel file
    """
    # Create DataFrame with results for all classes
    results_dict = {'Feature Name': feature_names}
    for class_name, scores in importance_scores_dict.items():
        results_dict[f'Importance_{class_name}'] = scores

    results_df = pd.DataFrame(results_dict)

    # Add ranking columns
    for class_name in importance_scores_dict.keys():
        score_col = f'Importance_{class_name}'
        rank_col = f'Rank_{class_name}'
        results_df[rank_col] = results_df[score_col].rank(ascending=False).astype(int)

    # Sort by combined importance in descending order
    if 'Combined' in importance_scores_dict:
        results_df = results_df.sort_values('Importance_Combined', ascending=False)

    # Save to Excel file
    results_df.to_excel(os.path.join(output_path, 'feature_importance_permutation_scores.xlsx'),
                        index=False)
    logging.info("Saved feature importance scores to Excel file")


def main():
    # Set input and output paths
    input_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '00_Data', 'Train.xlsx')  # Path to the input Excel file containing training data
    sheet_name = ''  # Name of the sheet in the Excel file (e.g., 'Sheet1')
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Results', 'Multicancer detection', '03_Ensemble')  # Directory path for saving output results

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load data
    logging.info("Loading data...")
    X, y, class_names, feature_names = load_data(input_file_path, sheet_name)

    logging.info(f"Loading completed. Feature list:")
    for i, name in enumerate(feature_names, 1):
        logging.info(f"{i}. {name}")

    # Genetic algorithm configuration
    ga_config = {
        'population_size': 40,
        'max_generations': 60,
        'crossover_rate': 0.8,
        'mutation_rate': 0.125,
        'elite_ratio': 0.125,
        'tournament_size': 3,
        'convergence_threshold': 10
    }

    # Configure ensemble algorithms
    config = {
        'output_path': output_path,
        'output_filename': 'Feature_Importance_results.xlsx',

        # Cross-validation configuration
        'outer_cv_splits': 5,
        'inner_cv_splits': 5,

        # Ensemble algorithms weights
        'xgb_weight': 0.6,
        'rf_weight': 0.4,

        # RF component configuration
        'rf_component': {
            'file_path': input_file_path,
            'sheet_name': sheet_name,
            'interaction_method': 'gaussian',
            'selection_method': 'boruta',
            'max_features': 40,
            'gaussian_gamma': 'auto',
            'scale_after_interaction': True,
            'debug_mode': True,
            'smote_sampling_strategy': 0.5,
            'smote_k_neighbors': 5,
            'outer_cv_splits': 5,
            'inner_cv_splits': 3,
            'ga_config': ga_config,
            'feature_params': {
                'interaction_method': 'gaussian',
                'selection_method': 'boruta',
                'max_features': 40,
                'gaussian_gamma': 'auto',
                'scale_after_interaction': True
            },
            'all_classes': class_names.tolist(),
        },

        # XGBoost component configuration
        'xgb_component': {
            'file_path': input_file_path,
            'sheet_name': sheet_name,
            'interaction_method': 'gaussian',
            'selection_method': 'mutual_info',
            'max_features': 40,
            'gaussian_gamma': 'auto',
            'scale_after_interaction': True,
            'debug_mode': True,
            'adasyn_sampling_strategy': 0.5,
            'adasyn_n_neighbors': 5,
            'outer_cv_splits': 5,
            'inner_cv_splits': 3,
            'ga_config': ga_config,
            'feature_params': {
                'interaction_method': 'gaussian',
                'selection_method': 'mutual_info',
                'max_features': 40,
                'gaussian_gamma': 'auto',
                'scale_after_interaction': True
            },
            'all_classes': class_names.tolist(),
        },

        # Color configuration
        'color_map': {
            class_names[0]: [254 / 255, 160 / 255, 64 / 255],
            class_names[1]: [242 / 255, 128 / 255, 128 / 255],
            class_names[2]: [88 / 255, 97 / 255, 172 / 255],
            class_names[3]: [106 / 255, 180 / 255, 193 / 255] if len(class_names) > 3 else [0, 0, 0]
        },
        'all_classes': class_names.tolist(),
    }

    # Initialize ensemble algorithms
    ensemble_model = EnsembleClassifier(config)

    logging.info("\nStarting permutation feature importance calculation...")

    # Store permutation feature importance for each class
    permutation_importance_scores_dict = {}

    # Calculate permutation feature importance for each class
    for i, class_name in enumerate(class_names):
        logging.info(f"\nCalculating permutation feature importance for class {class_name}...")
        importance_scores = calculate_permutation_importance(ensemble_model, X, y, i, feature_names, n_repeats=20)
        permutation_importance_scores_dict[class_name] = importance_scores

        # Print current class feature importance
        logging.info(f"\nPermutation feature importance ranking for class {class_name}:")
        sorted_indices = np.argsort(importance_scores)[::-1]
        for idx in sorted_indices:
            logging.info(f"{feature_names[idx]}: {importance_scores[idx]:.4f}")

    # Calculate combined feature importance (using weighted average)
    logging.info("\nCalculating combined permutation feature importance...")
    weights = np.array([(y == i).sum() / len(y) for i in range(len(class_names))])
    combined_perm_scores = np.zeros(len(feature_names))

    for i, scores in enumerate(permutation_importance_scores_dict.values()):
        combined_perm_scores += weights[i] * scores

    permutation_importance_scores_dict['Combined'] = combined_perm_scores

    # Print combined feature importance
    logging.info("\nCombined permutation feature importance ranking:")
    sorted_indices = np.argsort(combined_perm_scores)[::-1]
    for idx in sorted_indices:
        logging.info(f"{feature_names[idx]}: {combined_perm_scores[idx]:.4f}")

    # Save permutation importance results
    logging.info("\nSaving permutation feature importance results...")
    plot_feature_importance(permutation_importance_scores_dict, feature_names, output_path)
    save_importance_scores(permutation_importance_scores_dict, feature_names, output_path)

    logging.info(f"\nAnalysis results saved to folder: {output_path}")
    logging.info("- Per-class permutation feature importance charts: feature_importance_permutation_[class_name].png")
    logging.info("- Combined permutation feature importance chart: feature_importance_permutation_Combined.png")
    logging.info("- Permutation feature importance scores: feature_importance_permutation_scores.xlsx")


if __name__ == "__main__":
    main()