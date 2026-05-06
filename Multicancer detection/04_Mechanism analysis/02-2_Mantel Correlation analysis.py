import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def read_cancer_markers(file_path, sheet_name, marker_columns):
    """
    Read cancer marker data from specified sheet and columns

    Parameters:
    - file_path: Excel file path
    - sheet_name: Sheet name
    - marker_columns: List of marker column names
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        markers_df = df[marker_columns]
        return markers_df
    except Exception as e:
        print(f"Error reading {sheet_name}: {str(e)}")
        raise


def read_sensor_features(file_path, sheet_name):
    """
    Read sensor feature data from specified sheet
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        features = ['dint(9,1)', 'dint(8,3)', 'dint(7,5)', 'dint(7,3)', 'dint(6,5)', 'dint(6,4)',
                    'dwl(9,1)', 'dwl(8,3)', 'dwl(7,5)', 'dwl(7,3)', 'dwl(6,5)', 'dwl(6,4)']
        sensor_df = df[features]
        return sensor_df
    except Exception as e:
        print(f"Error reading sensor features from {sheet_name}: {str(e)}")
        raise


def handle_missing_values(cancer_markers, sensor_data):
    """
    Handle missing values by removing rows with any missing values
    in either cancer markers or sensor features datasets

    Returns cleaned datasets with same indices
    """
    # Combine datasets to find rows with missing values in either dataset
    combined = pd.concat([cancer_markers, sensor_data], axis=1)

    # Drop rows with any missing values
    cleaned = combined.dropna()

    # Print information about removed samples
    removed_count = len(cancer_markers) - len(cleaned)
    if removed_count > 0:
        print(f"   Removed {removed_count} samples with missing values")

    # Split back into separate dataframes but with same indices
    cleaned_markers = cleaned[cancer_markers.columns]
    cleaned_sensors = cleaned[sensor_data.columns]

    return cleaned_markers, cleaned_sensors


def calculate_mantel_test(matrix1, matrix2, n_permutations=999, method='spearman', random_seed=42):
    """
    Perform Mantel test between two matrices

    Parameters:
    - matrix1: First data matrix (usually tumor markers)
    - matrix2: Second data matrix (usually sensor features)
    - n_permutations: Number of permutations
    - method: Correlation method ('pearson' or 'spearman')
    - random_seed: Random seed for reproducibility

    Returns:
    - observed_corr: Observed correlation coefficient
    - p_value: P-value
    """
    # Set random seed
    np.random.seed(random_seed)

    # Select correlation function
    if method == 'spearman':
        corr_func = spearmanr
    elif method == 'pearson':
        corr_func = pearsonr
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pearson' or 'spearman'")

    # Calculate distance matrix (Euclidean distance)
    dist1 = squareform(pdist(matrix1, metric='euclidean'))
    dist2 = squareform(pdist(matrix2, metric='euclidean'))

    # Flatten distance matrix
    flat1 = dist1[np.triu_indices(dist1.shape[0], k=1)]
    flat2 = dist2[np.triu_indices(dist2.shape[0], k=1)]

    # Calculate correlation - Use Spearman (default) or Pearson
    observed_corr, _ = corr_func(flat1, flat2)

    # Permutation test
    permuted_corrs = []
    for i in range(n_permutations):
        # Use fixed seed for reproducibility
        np.random.seed(random_seed + i + 1)
        perm_flat2 = np.random.permutation(flat2)
        corr, _ = corr_func(flat1, perm_flat2)
        permuted_corrs.append(corr)

    # Calculate two-tailed p-value
    p_value = (sum(abs(pc) >= abs(observed_corr) for pc in permuted_corrs) + 1) / (n_permutations + 1)

    return observed_corr, p_value


def apply_fdr_correction(p_values, method='bh', alpha=0.05):
    """
    Apply FDR correction using Benjamini-Hochberg method

    Parameters:
    - p_values: array of p-values
    - method: 'bh' for Benjamini-Hochberg
    - alpha: significance level

    Returns:
    - adjusted_p_values: FDR-adjusted p-values
    - rejected: boolean array indicating significance
    """
    n = len(p_values)

    # Sort
    sorted_indices = np.argsort(p_values)
    sorted_p_values = np.array(p_values)[sorted_indices]

    # Benjamini-Hochberg FDR correction
    adjusted_p_values = np.zeros(n)
    adjusted_p_values[-1] = sorted_p_values[-1]

    for i in range(n - 2, -1, -1):
        adjusted_p_values[i] = min(
            sorted_p_values[i] * n / (i + 1),
            adjusted_p_values[i + 1]
        )

    adjusted_p_values = np.minimum(adjusted_p_values, 1.0)

    # Restore to original order
    final_adjusted_p = np.zeros(n)
    final_adjusted_p[sorted_indices] = adjusted_p_values

    # Determine significance
    rejected = final_adjusted_p < alpha

    return final_adjusted_p, rejected


def run_cancer_analysis(input_file_path, output_file_path, method='spearman',
                       n_permutations=999, fdr_threshold=0.05,
                       stratified=False, random_seed=42):
    """
    Run Mantel analysis for each cancer type against sensor features
    Use Spearman correlation (default)
    Include FDR multiple comparison correction

    Parameters:
    - method: 'spearman' (recommended) or 'pearson'
    - n_permutations: Number of permutations (default 999)
    - fdr_threshold: FDR significance threshold (default 0.05)
    - stratified: True=stratified correction, False=global correction
    - random_seed: Random seed
    """
    try:
        print("=" * 80)
        print(f"Mantel Test Analysis - {method.upper()} Correlation with FDR Correction")
        print("=" * 80)
        print(f"Method: {method.capitalize()} correlation")
        print(f"Permutations: {n_permutations}")
        print(f"FDR threshold: {fdr_threshold}")
        print(f"Correction strategy: {'Stratified (by cancer type)' if stratified else 'Global (all tests)'}")
        print(f"Random seed: {random_seed}")
        print("=" * 80)

        # Define cancer types, corresponding sheet names and markers
        cancer_configs = {
            'Liver Cancer': {
                'sheet': 'LC For Spearman',
                'markers': ['AFP', 'CEA', 'CA199']
            },
            'Lung Cancer': {
                'sheet': 'LuC For Spearman',
                'markers': ['CA153', 'CEA', 'NSE', 'CyFra21-1', 'SCC', 'HCG-β', 'proGRP', 'HE4']
            },
            'Ovarian Cancer': {
                'sheet': 'OC For Spearman',
                'markers': ['AFP', 'CA-125', 'HCG-β', 'CA-153', 'CA-724', 'NSE', 'HE4']
            }
        }

        # Sensor feature list
        sensor_features = ['dint(9,1)', 'dint(8,3)', 'dint(7,5)', 'dint(7,3)', 'dint(6,5)', 'dint(6,4)',
                           'dwl(9,1)', 'dwl(8,3)', 'dwl(7,5)', 'dwl(7,3)', 'dwl(6,5)', 'dwl(6,4)']

        results = []
        results_by_cancer = {}

        print()

        # Analyze each cancer type
        for cancer_type, config in cancer_configs.items():
            print(f"Processing {cancer_type}...")

            # Read cancer marker data
            cancer_markers = read_cancer_markers(
                input_file_path,
                config['sheet'],
                config['markers']
            )

            # Read sensor features from the same sheet
            sensor_data = read_sensor_features(
                input_file_path,
                config['sheet']
            )

            # Handle missing values
            print(f"   Original samples: {len(cancer_markers)}")
            cleaned_markers, cleaned_sensors = handle_missing_values(cancer_markers, sensor_data)
            print(f"   Cleaned samples: {len(cleaned_markers)}")

            # Sample count check
            if len(cleaned_markers) < 5:
                print(f"   WARNING: Only {len(cleaned_markers)} samples. Results may be unreliable!")

            # Store results for this cancer type
            cancer_results = []

            # Perform Mantel test for each sensor feature
            for feature in sensor_features:
                feature_data = cleaned_sensors[[feature]]

                # Calculate Mantel correlation
                cor, p_value = calculate_mantel_test(
                    cleaned_markers.values,
                    feature_data.values,
                    n_permutations=n_permutations,
                    method=method,
                    random_seed=random_seed
                )

                result = {
                    'Cancer Type': cancer_type,
                    'Sensor Feature': feature,
                    'Mantel r': round(cor, 4),
                    'P-value': round(p_value, 4),
                    'Sample Size': len(cleaned_markers),
                    'Method': method.capitalize()
                }

                results.append(result)
                cancer_results.append(result)

            results_by_cancer[cancer_type] = cancer_results
            print(f"   Completed {len(cancer_results)} tests")
            print()

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        print("=" * 80)
        print(f"Total tests performed: {len(results)}")
        print("=" * 80)

        # Apply FDR correction
        print(f"\nApplying FDR correction ({fdr_threshold} threshold)...")

        if stratified:
            # Stratified correction - Correct separately for each cancer type
            print("   Strategy: Stratified (per cancer type)")

            for cancer_type, cancer_results in results_by_cancer.items():
                # Extract p-values for this cancer type
                indices = [i for i, r in enumerate(results) if r['Cancer Type'] == cancer_type]
                p_values = [results[i]['P-value'] for i in indices]

                # FDR correction
                adjusted_p, rejected = apply_fdr_correction(p_values, alpha=fdr_threshold)

                # Update results
                for i, idx in enumerate(indices):
                    results[idx]['Adjusted P-value (FDR)'] = round(adjusted_p[i], 4)
                    results[idx]['FDR Significant'] = bool(rejected[i])

                sig_count = sum(rejected)
                print(f"      {cancer_type}: {sig_count}/{len(p_values)} significant")

        else:
            # Global correction - Correct all tests together
            print("   Strategy: Global (all tests together)")

            all_p_values = [r['P-value'] for r in results]
            adjusted_p, rejected = apply_fdr_correction(all_p_values, alpha=fdr_threshold)

            for i, result in enumerate(results):
                result['Adjusted P-value (FDR)'] = round(adjusted_p[i], 4)
                result['FDR Significant'] = bool(rejected[i])

            sig_count = sum(rejected)
            print(f"      Significant: {sig_count}/{len(all_p_values)}")

        # Update DataFrame
        results_df = pd.DataFrame(results)

        # Add significance markers
        def get_significance_mark(p, p_adj):
            """Significance markers based on FDR-corrected p-values"""
            if p_adj < 0.001:
                return "****"
            elif p_adj < 0.01:
                return "***"
            elif p_adj < 0.05:
                return "**"
            elif p_adj < 0.1:
                return "*"
            else:
                return ""

        results_df['Significance (FDR)'] = results_df.apply(
            lambda row: get_significance_mark(row['P-value'], row['Adjusted P-value (FDR)']),
            axis=1
        )

        # Reorder columns
        column_order = [
            'Cancer Type', 'Sensor Feature', 'Mantel r', 'Method',
            'P-value', 'Adjusted P-value (FDR)', 'Significance (FDR)', 'FDR Significant',
            'Sample Size'
        ]
        results_df = results_df[column_order]

        # Sort by cancer type and FDR p-value
        results_df = results_df.sort_values(['Cancer Type', 'Adjusted P-value (FDR)'])

        # Statistics
        sig_raw = sum(results_df['P-value'] < 0.05)
        sig_fdr = sum(results_df['FDR Significant'])

        print(f"\nResults Summary:")
        print(f"   Raw significant (p < 0.05): {sig_raw}/{len(results_df)} ({sig_raw/len(results_df)*100:.1f}%)")
        print(f"   FDR significant (FDR < {fdr_threshold}): {sig_fdr}/{len(results_df)} ({sig_fdr/len(results_df)*100:.1f}%)")
        print(f"   False positives removed: {sig_raw - sig_fdr}")

        # Ensure output directory exists
        output_path = Path(output_file_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results to Excel
        print(f"\nSaving results to Excel...")
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # Main results table
            results_df.to_excel(writer, sheet_name='Mantel Results', index=False)

            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Mantel Results']

            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'font_name': 'Arial',
                'font_size': 11,
                'align': 'center',
                'valign': 'vcenter',
                'bg_color': '#D9E1F2',
                'border': 1
            })

            cell_format = workbook.add_format({
                'font_name': 'Times New Roman',
                'font_size': 10,
                'align': 'center',
                'valign': 'vcenter',
                'border': 1
            })

            sig_format = workbook.add_format({
                'font_name': 'Times New Roman',
                'font_size': 10,
                'align': 'center',
                'valign': 'vcenter',
                'bg_color': '#FFF2CC',
                'border': 1
            })

            # Apply formats
            for col_num, value in enumerate(results_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                # Adjust width based on column content
                if 'P-value' in value or 'Mantel' in value:
                    worksheet.set_column(col_num, col_num, 16, cell_format)
                else:
                    worksheet.set_column(col_num, col_num, 18, cell_format)

            # Highlight FDR significant rows
            for row_num in range(1, len(results_df) + 1):
                if results_df.iloc[row_num - 1]['FDR Significant']:
                    for col_num in range(len(results_df.columns)):
                        cell_value = results_df.iloc[row_num - 1, col_num]
                        worksheet.write(row_num, col_num, cell_value, sig_format)

            # Create summary statistics table
            summary_data = []
            for cancer_type in cancer_configs.keys():
                cancer_subset = results_df[results_df['Cancer Type'] == cancer_type]

                summary_data.append({
                    'Cancer Type': cancer_type,
                    'Total Tests': len(cancer_subset),
                    'Raw Significant (p<0.05)': sum(cancer_subset['P-value'] < 0.05),
                    'FDR Significant': sum(cancer_subset['FDR Significant']),
                    'Mean Mantel r': round(cancer_subset['Mantel r'].mean(), 3),
                    'Std Mantel r': round(cancer_subset['Mantel r'].std(), 3),
                    'Mean Sample Size': int(cancer_subset['Sample Size'].mean())
                })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary by Cancer', index=False)

            # Format summary table
            summary_sheet = writer.sheets['Summary by Cancer']
            for col_num, value in enumerate(summary_df.columns.values):
                summary_sheet.write(0, col_num, value, header_format)
                summary_sheet.set_column(col_num, col_num, 18, cell_format)

        print(f"Results saved to: {output_path}")

        # Display FDR significant results
        if sig_fdr > 0:
            print(f"\nFDR Significant Results ({sig_fdr} total):")
            print("=" * 80)
            significant_results = results_df[results_df['FDR Significant']]
            for _, row in significant_results.iterrows():
                print(f"   {row['Cancer Type']:20s} | {row['Sensor Feature']:12s} | "
                      f"r={row['Mantel r']:7.4f} | p(FDR)={row['Adjusted P-value (FDR)']:7.4f} {row['Significance (FDR)']}")

        print("\n" + "=" * 80)
        print("Analysis Complete!")
        print("=" * 80)
        print(f"Method: {method.capitalize()} Mantel correlation")
        print(f"FDR correction: {'Stratified' if stratified else 'Global'}")
        print(f"Output: {output_path}")
        print("=" * 80)

        return results_df

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """
    Main function - Mantel test analysis
    Use Spearman correlation (more robust)
    Include FDR multiple comparison correction
    """
    # Configuration parameters
    config = {
        'input_file': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Source data for Mechanism analysis.xlsx'),  # Path to the input Excel file containing training data
        'output_file': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Results', 'Multicancer detection', '04_Mechanism analysis', 'mantel_correlation_analysis.xlsx'),  # Output file path for saving results

        # Mantel test parameters
        'method': 'spearman',          # 'spearman' (recommended) or 'pearson'
        'n_permutations': 9999,        # Number of permutations (recommend at least 999)

        # FDR correction parameters
        'fdr_threshold': 0.05,         # FDR significance threshold
        'stratified': False,           # True=stratified correction, False=global correction

        # Random seed
        'random_seed': 42
    }

    # Run analysis
    results = run_cancer_analysis(
        input_file_path=config['input_file'],
        output_file_path=config['output_file'],
        method=config['method'],
        n_permutations=config['n_permutations'],
        fdr_threshold=config['fdr_threshold'],
        stratified=config['stratified'],
        random_seed=config['random_seed']
    )

    return results


if __name__ == "__main__":
    main()