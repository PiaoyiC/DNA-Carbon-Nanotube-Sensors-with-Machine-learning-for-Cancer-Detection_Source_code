import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def analyze_single_sheet_correlation(input_file, sheet_name, r_threshold=0.7, p_threshold=0.05,
                                     correction_method='fdr_bh'):
    """
    Analyze correlations for a single sheet with Spearman correlation and multiple comparison correction

    Parameters:
    input_file: Excel file path
    sheet_name: Name of the sheet to analyze
    r_threshold: Correlation coefficient threshold
    p_threshold: Significance level threshold (applied after correction)
    correction_method: Multiple comparison correction method ('fdr_bh', 'bonferroni', 'holm', 'none')
    """
    # Define sensor features
    features = ['dint(9,1)', 'dint(8,3)', 'dint(7,5)', 'dint(7,3)', 'dint(6,5)', 'dint(6,4)',
                'dwl(9,1)', 'dwl(8,3)', 'dwl(7,5)', 'dwl(7,3)', 'dwl(6,5)', 'dwl(6,4)']

    # Read data
    print(f"\nReading data from sheet: {sheet_name}")
    data = pd.read_excel(input_file, sheet_name=sheet_name)[features]

    # Check missing values
    missing_info = data.isnull().sum()
    if missing_info.sum() > 0:
        print(f"Missing values detected:\n{missing_info[missing_info > 0]}")
        print("Removing rows with missing values...")
        data = data.dropna()
        print(f"Final sample size: {len(data)}")

    # Calculate Spearman correlation matrix
    correlation_matrix = data.corr(method='spearman')

    # Calculate p-value matrix
    n_features = len(features)
    p_matrix_raw = pd.DataFrame(np.zeros((n_features, n_features)),
                                columns=features,
                                index=features)
    p_matrix_corrected = pd.DataFrame(np.zeros((n_features, n_features)),
                                      columns=features,
                                      index=features)

    # Store significant feature pairs
    significant_pairs = []
    all_comparisons = []

    # Calculate p-values and collect all comparison LOO results
    for i in range(n_features):
        for j in range(i + 1, n_features):
            rho, p = stats.spearmanr(data.iloc[:, i], data.iloc[:, j])
            p_matrix_raw.iloc[i, j] = p
            p_matrix_raw.iloc[j, i] = p

            all_comparisons.append({
                'i': i,
                'j': j,
                'Feature1': features[i],
                'Feature2': features[j],
                'Spearman_rho': rho,
                'P_value_raw': p
            })

    # Apply multiple comparison correction
    if correction_method != 'none':
        p_values = [comp['P_value_raw'] for comp in all_comparisons]

        print(f"\nApplying {correction_method} correction for {len(p_values)} comparisons...")
        print(f"Original alpha level: {p_threshold}")

        rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            p_values, alpha=p_threshold, method=correction_method
        )

        # Update comparison LOO results and build corrected p-value matrix
        for idx, comp in enumerate(all_comparisons):
            comp['P_value_corrected'] = p_corrected[idx]
            comp['Significant'] = rejected[idx]

            # Fill corrected p-value matrix
            i, j = comp['i'], comp['j']
            p_matrix_corrected.iloc[i, j] = p_corrected[idx]
            p_matrix_corrected.iloc[j, i] = p_corrected[idx]

        if correction_method == 'bonferroni':
            print(f"Bonferroni corrected alpha: {alpha_bonf:.6f}")
        elif correction_method == 'fdr_bh':
            print(f"FDR (Benjamini-Hochberg) correction applied")
        elif correction_method == 'holm':
            print(f"Holm-Bonferroni sequential correction applied")

        n_significant_raw = sum(p < p_threshold for p in p_values)
        n_significant_corrected = sum(rejected)
        print(f"Raw significant pairs: {n_significant_raw}")
        print(f"Corrected significant pairs: {n_significant_corrected}")

    else:

        for comp in all_comparisons:
            comp['P_value_corrected'] = comp['P_value_raw']
            comp['Significant'] = comp['P_value_raw'] < p_threshold

            i, j = comp['i'], comp['j']
            p_matrix_corrected.iloc[i, j] = comp['P_value_raw']
            p_matrix_corrected.iloc[j, i] = comp['P_value_raw']

    # Filter significant feature pairs
    for comp in all_comparisons:
        if abs(comp['Spearman_rho']) >= r_threshold and comp['Significant']:
            correlation_strength = "Very Strong" if abs(comp['Spearman_rho']) >= 0.8 else "Strong"

            pair_info = {
                'Feature1': comp['Feature1'],
                'Feature2': comp['Feature2'],
                'Spearman_rho': round(comp['Spearman_rho'], 3),
                'P_value_raw': round(comp['P_value_raw'], 6),
                'P_value_corrected': round(comp['P_value_corrected'], 6),
                'Significant': comp['Significant'],
                'Strength': correlation_strength
            }

            significant_pairs.append(pair_info)

    # Convert to DataFrame and sort by absolute correlation coefficient
    if significant_pairs:
        significant_df = pd.DataFrame(significant_pairs)
        significant_df = significant_df.sort_values(by='Spearman_rho', key=abs, ascending=False)
    else:
        significant_df = pd.DataFrame(
            columns=['Feature1', 'Feature2', 'Spearman_rho', 'P_value_raw', 'P_value_corrected', 'Significant',
                     'Strength'])

    # Create DataFrame for all comparison LOO results for saving
    all_results_df = pd.DataFrame(all_comparisons)
    if not all_results_df.empty:
        all_results_df = all_results_df.sort_values(by='P_value_corrected')
        all_results_df['Spearman_rho'] = all_results_df['Spearman_rho'].round(3)
        all_results_df['P_value_raw'] = all_results_df['P_value_raw'].round(6)
        all_results_df['P_value_corrected'] = all_results_df['P_value_corrected'].round(6)
        all_results_df = all_results_df.drop(['i', 'j'], axis=1)

    return correlation_matrix, p_matrix_raw, p_matrix_corrected, significant_df, all_results_df


def get_correction_description(method):
    """Get description for correction method"""
    descriptions = {
        'fdr_bh': 'Benjamini-Hochberg False Discovery Rate correction - Controls expected proportion of false discoveries',
        'bonferroni': 'Bonferroni correction - Controls family-wise error rate by dividing alpha by number of tests',
        'holm': 'Holm-Bonferroni sequential correction - Less conservative than Bonferroni',
        'none': 'No multiple comparison correction applied - Uses raw p-values'
    }
    return descriptions.get(method, 'Unknown correction method')


def get_correction_reference(method):
    """Get reference for correction method"""
    references = {
        'fdr_bh': 'Benjamini & Hochberg (1995). Journal of the Royal Statistical Society B, 57(1), 289-300',
        'bonferroni': 'Bonferroni (1936). Teoria statistica delle classi e calcolo delle probabilità',
        'holm': 'Holm (1979). Scandinavian Journal of Statistics, 6(2), 65-70',
        'none': 'No reference - standard significance testing'
    }
    return references.get(method, 'No reference available')


def save_results(correlation_matrix, p_matrix_raw, p_matrix_corrected, significant_df, all_results_df, output_file,
                 sheet_name, correction_method):
    """Save analysis LOO results to Excel file"""
    with pd.ExcelWriter(output_file) as writer:
        # Save Spearman correlation matrix
        correlation_matrix.round(3).to_excel(writer, sheet_name=f'{sheet_name}_Spearman_Corr')

        # Save raw p-value matrix
        p_matrix_raw.round(6).to_excel(writer, sheet_name=f'{sheet_name}_P_Values_Raw')

        # Save corrected p-value matrix
        p_matrix_corrected.round(6).to_excel(writer, sheet_name=f'{sheet_name}_P_Values_Corrected')

        # Save all comparison LOO results (including correction information)
        all_results_df.to_excel(writer, sheet_name=f'{sheet_name}_All_Results', index=False)

        # Save significant feature pairs
        significant_df.to_excel(writer, sheet_name=f'{sheet_name}_Sig_Pairs', index=False)

        # Add method information sheet
        method_info = pd.DataFrame({
            'Parameter': ['Correlation Method', 'Correction Method', 'Description', 'Reference'],
            'Value': [
                'Spearman Rank Correlation',
                correction_method,
                get_correction_description(correction_method),
                get_correction_reference(correction_method)
            ]
        })
        method_info.to_excel(writer, sheet_name=f'{sheet_name}_Method_Info', index=False)


def main():
    # Configuration parameters
    config = {
        'input_file': r"",  # Path to the input Excel file containing training data
        'output_file': r'',  # Directory path for saving output LOO results
        'sheet_name': 'All',
        'r_threshold': 0.7,  # Correlation coefficient threshold
        'p_threshold': 0.05,  # P-value threshold
        'correction_method': 'fdr_bh'  # Multiple comparison correction method: 'fdr_bh', 'bonferroni', 'holm', 'none'
    }

    # Run analysis
    print(f"Analyzing Spearman correlations for sheet: {config['sheet_name']}")
    print(f"Using correction method: {config['correction_method']}")

    correlation_matrix, p_matrix_raw, p_matrix_corrected, significant_pairs, all_results = analyze_single_sheet_correlation(
        config['input_file'],
        config['sheet_name'],
        config['r_threshold'],
        config['p_threshold'],
        config['correction_method']
    )

    # Save LOO results
    print("\nSaving LOO results...")
    save_results(
        correlation_matrix,
        p_matrix_raw,
        p_matrix_corrected,
        significant_pairs,
        all_results,
        config['output_file'],
        config['sheet_name'],
        config['correction_method']
    )

    # Print LOO results summary
    print(f"\nResults saved to {config['output_file']}")
    print(f"\nAnalysis Summary:")
    print(f"- Correlation method: Spearman rank correlation")
    print(f"- Correction method: {config['correction_method']}")
    print(f"- Correlation threshold: |ρ| >= {config['r_threshold']}")
    print(f"- Significance threshold: p < {config['p_threshold']} (after correction)")
    print(f"- Total comparisons: {len(all_results)}")
    print(f"- Significant correlations found: {len(significant_pairs)}")

    # If there are significant feature pairs, print them
    if not significant_pairs.empty:
        print("\nSignificant correlations (after correction):")
        display_cols = ['Feature1', 'Feature2', 'Spearman_rho', 'P_value_corrected', 'Strength']
        print(significant_pairs[display_cols].to_string(index=False))
    else:
        print("\nNo significant correlations found after multiple comparison correction.")
        print("Consider:")
        print("1. Using a less conservative correction method (e.g., 'fdr_bh' instead of 'bonferroni')")
        print("2. Lowering the correlation threshold")
        print("3. Using 'none' for correction to see raw LOO results")


if __name__ == "__main__":
    main()