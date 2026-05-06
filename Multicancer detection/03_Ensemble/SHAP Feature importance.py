import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.preprocessing import LabelEncoder
import logging
import sys
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')


# Import module directly from file
def import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import EnsembleClassifier from file
current_dir = os.path.dirname(os.path.abspath(__file__))
ensemble_file = os.path.join(current_dir, "Ensemble_RF+XGB_CV.py")
ensemble_module = import_from_file("ensemble_model", ensemble_file)
EnsembleClassifier = ensemble_module.EnsembleClassifier


def load_data(file_path, sheet_name):
    """Load data"""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # Extract sample type and features
    X = df.iloc[:, 1:14]  # Columns 2-14 as features
    y = df.iloc[:, 0]  # Column 1 as labels
    return X, y, sorted(y.unique())


def create_shap_plot(shap_values, X, feature_names, class_name, output_path):
    """Create SHAP visualization plot"""
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.sans-serif': ['Arial'],
        'font.serif': ['Arial'],
        'font.monospace': ['Arial'],
        'font.size': 24,
        'axes.linewidth': 3,
        'text.color': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.labelcolor': 'black',
        'axes.titlecolor': 'black',
        'svg.fonttype': 'none',
        'mathtext.fontset': 'cm',
        'mathtext.default': 'regular',
        'axes.unicode_minus': False
    })

    plt.clf()
    plt.figure(figsize=(12, 8))

    # Create DataFrame with feature names and original data
    X_with_names = pd.DataFrame(X, columns=feature_names)

    # Use correct parameter settings
    shap.summary_plot(
        shap_values,
        X_with_names,
        plot_type="dot",
        show=False,
        max_display=12,
        alpha=0.5,
        plot_size=(12, 8)
    )

    for ax in plt.gcf().get_axes():
        for collection in ax.collections:
            try:
                collection.set_sizes([100] * len(collection.get_offsets()))
            except:
                continue

    ax = plt.gca()

    for spine in ax.spines.values():
        spine.set_linewidth(5)

    ax.tick_params(width=5, length=10)

    plt.title(f'SHAP Values Summary - {class_name}',
              pad=24,
              fontsize=24,
              fontname='Arial')
    plt.xlabel('SHAP value (impact on algorithms output)',
               fontsize=24,
               fontname='Arial')

    plt.yticks(fontsize=24)

    plt.xticks(fontsize=24)

    for text_obj in plt.gcf().findobj(plt.Text):
        text_obj.set_fontname('Arial')

    try:

        for child in ax.get_children():
            if isinstance(child, plt.matplotlib.colorbar.Colorbar):

                child.ax.set_box_aspect(30)
                child.outline.set_linewidth(5)

                child.ax.tick_params(labelsize=32)

                child.ax.set_ylabel('Feature value',
                                    fontsize=32,
                                    fontname='Arial',
                                    fontweight='bold')

                child.ax.tick_params(labelsize=32)

                child.ax.yaxis.label.set_size(32)
                child.ax.yaxis.label.set_fontname('Arial')

                if len(child.ax.texts) > 0:
                    for text in child.ax.texts:
                        text.set_fontsize(32)
                        text.set_fontname('Arial')
                break
    except:
        pass

    plt.tight_layout()
    output_file_svg = os.path.join(output_path, f'shap_summary_{class_name}.svg')

    plt.savefig(
        output_file_svg,
        format='svg',
        bbox_inches='tight',
        pad_inches=0.5
    )

    plt.close()

    logging.info(f"SHAP summary plot (SVG) saved: {output_file_svg}")


def create_force_plot(shap_values, X, feature_names, class_name, output_path, sample_idx, is_positive, explainer):
    """Create SHAP force plot

    Parameters:
        shap_values: SHAP values
        X: Sample feature values
        feature_names: List of feature names
        class_name: Class name
        output_path: Output path
        sample_idx: Sample index
        is_positive: Whether it is a positive sample
        explainer: SHAP explainer object (used to get the correct base_value)
    """

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.sans-serif': ['Arial'],
        'font.serif': ['Arial'],
        'font.monospace': ['Arial'],
        'font.size': 24,
        'axes.linewidth': 3,
        'text.color': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.labelcolor': 'black',
        'axes.titlecolor': 'black',
        'svg.fonttype': 'none',
        'mathtext.fontset': 'cm',
        'mathtext.default': 'regular',
        'axes.unicode_minus': False
    })

    plt.figure(figsize=(24, 4))

    feature_df = pd.DataFrame([X], columns=feature_names)

    base_value = explainer.expected_value

    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    print(f"   Class {class_name}, Sample #{sample_idx}: Correct base_value = {base_value:.6f}")

    shap_sum = np.sum(shap_values)
    predicted_value = base_value + shap_sum
    print(f"   Verification: base_value({base_value:.3f}) + shap_sum({shap_sum:.3f}) = {predicted_value:.3f}")

    shap.force_plot(
        base_value=base_value,
        shap_values=shap_values,
        features=feature_df,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
        figsize=(24, 4),
        text_rotation=0,
        contribution_threshold=0.0
    )

    for text_obj in plt.gcf().findobj(plt.Text):

        text_obj.set_fontname('Arial')

        content = text_obj.get_text()
        if any(c.isdigit() for c in content):

            try:

                if "=" in content:
                    parts = content.split("=")
                    feature_name = parts[0].strip()
                    value_str = parts[1].strip()

                    try:
                        value = float(value_str)
                        new_text = f"{feature_name} = {value:.3f}"
                        text_obj.set_text(new_text)
                    except ValueError:
                        pass

                elif "." in content and not "f(x)" in content:
                    try:
                        value = float(content)
                        text_obj.set_text(f"{value:.3f}")
                    except ValueError:
                        pass
            except Exception as e:
                logging.debug(f"Error processing text: {str(e)}")

    sample_type = "Positive Sample" if is_positive else "Negative Sample"
    plt.title(f"{class_name} - {sample_type} (Sample #{sample_idx})", fontsize=24, fontname='Arial')

    sample_type_str = "positive" if is_positive else "negative"
    output_file_svg = os.path.join(output_path, f"force_plot_{class_name}_{sample_type_str}_sample{sample_idx}.svg")

    plt.savefig(
        output_file_svg,
        format='svg',
        bbox_inches='tight',
        pad_inches=0.1
    )

    plt.close()

    force_plot_data = {
        'Class': class_name,
        'Sample_Index': sample_idx,
        'Is_Positive': is_positive,
        'Base_Value': float(f"{base_value:.3f}"),
        'Feature': feature_names,
        'Feature_Value': [float(f"{val:.3f}") for val in X],
        'SHAP_Value': [float(f"{val:.3f}") for val in shap_values]
    }

    logging.info(f"SVG format force plot saved: {output_file_svg}")

    return force_plot_data


class FeatureProcessor:
    """Feature processing class for handling original features and mapping SHAP values"""

    def __init__(self, rf_feature_interaction, xgb_feature_interaction, rf_model, xgb_model, xgb_weight):
        self.rf_feature_interaction = rf_feature_interaction
        self.xgb_feature_interaction = xgb_feature_interaction
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.xgb_weight = xgb_weight

    def process_and_predict(self, X):
        """Process features and predict"""
        # Process RF features
        X_rf = self.rf_feature_interaction.transform(X)
        # Process XGB features
        X_xgb = self.xgb_feature_interaction.transform(X)

        # Individual algorithms predictions
        rf_proba = self.rf_model.predict_proba(X_rf)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X_xgb)[:, 1]

        # Ensemble prediction
        ensemble_proba = (1 - self.xgb_weight) * rf_proba + self.xgb_weight * xgb_proba

        return ensemble_proba


def train_ensemble_for_class(X, y, class_name, config):
    """
    Train ensemble algorithms for specific class - Using original optimization process
    """
    print(f"Training ensemble algorithms for class {class_name}...")

    # Create binary classification labels
    y_binary = (y == class_name).astype(int)

    # Use the complete training process of the original EnsembleClassifier
    ensemble = EnsembleClassifier(config)

    # Load data
    rf_data, xgb_data = ensemble.load_data()

    # Get feature processors
    rf_feature_interaction = ensemble.rf_component.feature_interaction
    xgb_feature_interaction = ensemble.xgb_component.feature_interaction

    # Data augmentation
    rf_balanced_data, xgb_balanced_data = ensemble.apply_data_augmentation(rf_data, xgb_data, class_name)

    # Training data preparation
    rf_X_train = rf_balanced_data[0]
    rf_y_train = rf_balanced_data[1]
    xgb_X_train = xgb_balanced_data[0]
    xgb_y_train = xgb_balanced_data[1]

    # GA optimize RF algorithms hyperparameters
    fold_idx = 0
    rf_best_params = ensemble.rf_component._optimize_hyperparameters(
        rf_X_train, rf_y_train, class_name, fold_idx
    )
    rf_model = ensemble.rf_component._create_model(rf_best_params)
    rf_model.fit(rf_X_train, rf_y_train)

    # GA optimize XGBoost algorithms hyperparameters
    xgb_best_params = ensemble.xgb_component._optimize_hyperparameters(
        xgb_X_train, xgb_y_train, class_name, fold_idx
    )
    xgb_model = ensemble.xgb_component._create_model(xgb_best_params)
    xgb_model.fit(xgb_X_train, xgb_y_train)

    # Optimize ensemble weight
    rf_y_pred_proba = rf_model.predict_proba(rf_X_train)[:, 1]
    xgb_y_pred_proba = xgb_model.predict_proba(xgb_X_train)[:, 1]
    best_ensemble_weight = ensemble._optimize_ensemble_weight(
        rf_y_pred_proba, xgb_y_pred_proba, rf_y_train
    )

    logging.info(
        f"Ensemble algorithms training completed for class {class_name}, XGBoost weight: {best_ensemble_weight:.2f}")

    # Create feature processor
    feature_processor = FeatureProcessor(
        rf_feature_interaction,
        xgb_feature_interaction,
        rf_model,
        xgb_model,
        best_ensemble_weight
    )

    return feature_processor, rf_feature_interaction, xgb_feature_interaction


def select_samples_for_force_plot(X, y, class_name, feature_processor):
    """Select samples for force plot - Ensure selecting correctly classified samples

    Parameters:
        X: Feature data
        y: Label data
        class_name: Class name
        feature_processor: Feature processor

    Returns:
        Positive sample indices, negative sample indices
    """
    # Create binary labels
    y_binary = (y == class_name).astype(int)

    # Use algorithms to predict all samples
    X_values = X.values
    probabilities = feature_processor.process_and_predict(X_values)
    predictions = (probabilities >= 0.5).astype(int)

    # Find correctly classified samples
    correct_predictions = (predictions == y_binary)

    # Find correctly classified positive samples
    positive_indices = np.where((y_binary == 1) & correct_predictions)[0]
    negative_indices = np.where((y_binary == 0) & correct_predictions)[0]

    # Select the first correctly classified positive and negative samples
    positive_idx = positive_indices[0] if len(positive_indices) > 0 else None
    negative_idx = negative_indices[0] if len(negative_indices) > 0 else None

    if positive_idx is not None:
        print(
            f"   Selected positive sample#{positive_idx}: True={y_binary[positive_idx]}, Predicted prob={probabilities[positive_idx]:.3f}, Predicted={predictions[positive_idx]} ✓")

    if negative_idx is not None:
        print(
            f"   Selected negative sample#{negative_idx}: True={y_binary[negative_idx]}, Predicted prob={probabilities[negative_idx]:.3f}, Predicted={predictions[negative_idx]} ✓")

    return positive_idx, negative_idx


def analyze_shap_importance(X, y, class_names, output_path, config):
    """Calculate SHAP importance and save results"""
    # Get original feature names
    original_feature_names = X.columns
    all_results = {
        'Feature': list(original_feature_names),
        'Mean_Importance': np.zeros(len(original_feature_names))
    }

    # Initialize result columns for each class
    for class_name in class_names:
        all_results[f'Importance_{class_name}'] = np.zeros(len(original_feature_names))

    # Save force plot data
    force_plot_data_all = []

    # Keep original feature data
    X_values = X.values

    for class_name in class_names:
        print(f"\nAnalyzing class: {class_name}")

        # Train ensemble algorithms for current class
        feature_processor, rf_feature_interaction, xgb_feature_interaction = train_ensemble_for_class(X, y, class_name,
                                                                                                      config)

        # Use wrapper function to process features and predict
        def model_predict(X_input):
            return feature_processor.process_and_predict(X_input)

        # Calculate SHAP values
        print(f"Calculating SHAP values for class {class_name}...")

        # Create background dataset
        background = shap.sample(X_values, 100, random_state=42)

        # Create SHAP explainer - Use KernelExplainer as it can handle any black-box algorithms
        explainer = shap.KernelExplainer(model_predict, background)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_values)

        # Print base_value information
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value_display = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value_display = base_value
        print(f"   Correct base_value for class {class_name}: {base_value_display:.6f}")

        # Create SHAP plot
        create_shap_plot(shap_values, X_values, original_feature_names, class_name, output_path)

        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)

        # Store results
        all_results[f'Importance_{class_name}'] = feature_importance
        all_results['Mean_Importance'] += feature_importance

        # Select samples for force plot
        positive_idx, negative_idx = select_samples_for_force_plot(X, y, class_name, feature_processor)

        # Create force plot - Positive sample
        if positive_idx is not None:
            # Get SHAP values for single sample
            sample_shap_values = shap_values[positive_idx]
            sample_features = X_values[positive_idx]

            # Create force plot and get data - Pass explainer parameter
            force_data_positive = create_force_plot(
                sample_shap_values,
                sample_features,
                original_feature_names,
                class_name,
                output_path,
                positive_idx,
                True,
                explainer  # Pass explainer to get correct base_value
            )

            # Convert to DataFrame and add to list
            for i, feature in enumerate(original_feature_names):
                force_plot_data_all.append({
                    'Class': class_name,
                    'Sample_Index': positive_idx,
                    'Is_Positive': True,
                    'Feature': feature,
                    'Feature_Value': float(f"{sample_features[i]:.3f}"),
                    'SHAP_Value': float(f"{sample_shap_values[i]:.3f}"),
                    'Base_Value': float(f"{force_data_positive['Base_Value']:.3f}")
                })

        # Create force plot - Negative sample
        if negative_idx is not None:
            # Get SHAP values for single sample
            sample_shap_values = shap_values[negative_idx]
            sample_features = X_values[negative_idx]

            # Create force plot and get data - Pass explainer parameter
            force_data_negative = create_force_plot(
                sample_shap_values,
                sample_features,
                original_feature_names,
                class_name,
                output_path,
                negative_idx,
                False,
                explainer  # Pass explainer to get correct base_value
            )

            # Convert to DataFrame and add to list
            for i, feature in enumerate(original_feature_names):
                force_plot_data_all.append({
                    'Class': class_name,
                    'Sample_Index': negative_idx,
                    'Is_Positive': False,
                    'Feature': feature,
                    'Feature_Value': float(f"{sample_features[i]:.3f}"),
                    'SHAP_Value': float(f"{sample_shap_values[i]:.3f}"),
                    'Base_Value': float(f"{force_data_negative['Base_Value']:.3f}")
                })

        # Print feature importance ranking
        print(f"\nFeature importance ranking for class {class_name}:")
        feature_importance_pairs = list(zip(original_feature_names, feature_importance))
        sorted_pairs = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)

        for feature, importance in sorted_pairs:
            print(f"{feature}: {importance:.4f}")

    # Calculate average importance
    all_results['Mean_Importance'] /= len(class_names)

    # Create and save result DataFrame
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('Mean_Importance', ascending=False)

    # Create force plot data DataFrame
    force_plot_df = pd.DataFrame(force_plot_data_all)

    # Save to Excel - With multiple sheets
    excel_path = os.path.join(output_path, 'ensemble_shap_importance_results.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # First sheet - Feature importance
        results_df.to_excel(writer, sheet_name='Feature_Importance', index=False, float_format='%.6f')

        # Second sheet - Force plot data
        force_plot_df.to_excel(writer, sheet_name='Force_Plot_Data', index=False, float_format='%.3f')

    return results_df


def main():
    # Set paths
    input_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '00_Data', 'Train.xlsx'),  # Path to the input Excel file containing training data
    sheet_name = '',  # Name of the sheet in the Excel file (e.g., 'Sheet1')
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Results', 'Multicancer detection', '03_Ensemble')  # Directory path for saving output results

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load data
    print("Loading data...")
    X, y, class_names = load_data(input_file_path, sheet_name)
    print(f"Data loading completed, number of features: {X.shape[1]}")
    print(f"Classes: {class_names}")

    # Genetic algorithm configuration
    ga_config = {
        'population_size': 50,
        'max_generations': 60,
        'crossover_rate': 0.8,
        'mutation_rate': 0.125,
        'elite_ratio': 0.125,
        'tournament_size': 3,
        'convergence_threshold': 10
    }

    # Create configuration
    config = {
        'output_path': output_path,
        'output_filename': 'Ensemble_SHAP_results.xlsx',

        # Cross-validation configuration
        'outer_cv_splits': 5,
        'inner_cv_splits': 5,

        # Ensemble algorithms weights
        'xgb_weight': 0.6,  # Initial XGBoost weight
        'rf_weight': 0.4,  # Initial RF weight

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
            'inner_cv_splits': 5,
            'ga_config': ga_config,
            'feature_params': {
                'interaction_method': 'gaussian',
                'selection_method': 'boruta',
                'max_features': 40,
                'gaussian_gamma': 'auto',
                'scale_after_interaction': True
            },
            'all_classes': class_names
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
            'inner_cv_splits': 5,
            'ga_config': ga_config,
            'feature_params': {
                'interaction_method': 'gaussian',
                'selection_method': 'mutual_info',
                'max_features': 40,
                'gaussian_gamma': 'auto',
                'scale_after_interaction': True
            },
            'all_classes': class_names,
        },

        # Color configuration
        'color_map': {
            'LC': [254 / 255, 160 / 255, 64 / 255],
            'LuC': [242 / 255, 128 / 255, 128 / 255],
            'OC': [88 / 255, 97 / 255, 172 / 255],
            'N': [106 / 255, 180 / 255, 193 / 255]
        },

        'all_classes': class_names,
    }

    # Perform SHAP analysis and save results
    print("\nStarting SHAP analysis")
    print("=" * 80)
    results_df = analyze_shap_importance(X, y, class_names, output_path, config)

    print(f"\nAnalysis results saved to folder: {output_path}")
    print("Output files:")
    print("- Per-class SHAP analysis plots: shap_summary_[class_name].svg")
    print("- Per-class SHAP force plots: force_plot_[class_name]_[positive/negative]_sample[index].svg")
    print("- Feature importance results: ensemble_shap_importance_results.xlsx")
    print("  - Sheet 1: Feature_Importance - Feature importance values")
    print("  - Sheet 2: Force_Plot_Data - Force plot data")

    # Print top 5 most important features
    print("\nTop 5 overall most important features:")
    top_features = results_df.head(5)
    for _, row in top_features.iterrows():
        print(f"   {row['Feature']}: {row['Mean_Importance']:.4f}")


if __name__ == "__main__":
    main()