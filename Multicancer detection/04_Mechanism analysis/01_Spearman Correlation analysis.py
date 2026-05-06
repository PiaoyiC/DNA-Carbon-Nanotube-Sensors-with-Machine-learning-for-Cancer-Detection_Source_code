import pandas as pd
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.colors as mcolors
from matplotlib import font_manager
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")  # Ignore warnings

def analyze_tumor_markers_correlation(input_file, output_file, p_threshold=0.1, dpi=600):
    """
    Analyze Spearman correlation between tumor markers of multiple tumor types and fluorescence features
    """
    print(f"Analyzing file: {input_file}")

    try:
        # Use Path object to handle paths
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"File does not exist: {input_path.absolute()}")

        print(f"File found: {input_path.absolute()}")

        sheet_markers = {
            'LC For Spearman': ['AFP', 'CEA', 'CA199'],
            'LuC For Spearman': ['CA153', 'CEA', 'NSE', 'CyFra21-1', 'SCC', 'HCG-β', 'proGRP', 'HE4'],
            'OC For Spearman': ['AFP', 'CA-125', 'HCG-β', 'CA-153', 'CA-724', 'NSE', 'HE4']
        }

        # All fluorescence features
        fluorescence_features = [
            'dint(9,1)', 'dint(8,3)', 'dint(7,5)', 'dint(7,3)', 'dint(6,5)', 'dint(6,4)',
            'dwl(9,1)', 'dwl(8,3)', 'dwl(7,5)', 'dwl(7,3)', 'dwl(6,5)', 'dwl(6,4)'
        ]

        # Initialize summary results list
        all_results = []

        # Read all sheet names from Excel file
        try:
            excel_file = pd.ExcelFile(input_path)
            available_sheets = excel_file.sheet_names
            print(f"Available sheets in file: {available_sheets}")

            # Check if requested sheets exist
            for sheet_name in sheet_markers.keys():
                if sheet_name not in available_sheets:
                    print(f"Warning: Sheet '{sheet_name}' not found in Excel file.")

            # Filter out non-existent sheets
            sheet_markers = {sheet: markers for sheet, markers in sheet_markers.items()
                            if sheet in available_sheets}

            if not sheet_markers:
                raise ValueError("None of the requested sheets were found in the Excel file.")
        except Exception as e:
            print(f"Error reading Excel file sheets: {str(e)}")
            raise

        # Process each sheet
        for sheet_name, tumor_markers in sheet_markers.items():
            print(f"Processing sheet: {sheet_name}")

            try:
                # Read data
                data = pd.read_excel(input_path, sheet_name=sheet_name)
                print(f"Data columns in {sheet_name}: {data.columns.tolist()}")

                # Check if required columns exist
                missing_tumor_markers = [col for col in tumor_markers if col not in data.columns]
                missing_features = [col for col in fluorescence_features if col not in data.columns]

                if missing_tumor_markers:
                    print(f"Warning: The following tumor markers are missing in {sheet_name}: {', '.join(missing_tumor_markers)}")
                    tumor_markers = [marker for marker in tumor_markers if marker not in missing_tumor_markers]

                if missing_features:
                    print(f"Warning: The following fluorescence features are missing in {sheet_name}: {', '.join(missing_features)}")
                    fluorescence_features = [feature for feature in fluorescence_features if feature not in missing_features]

                if not tumor_markers or not fluorescence_features:
                    print(f"Error: No valid columns found in {sheet_name}. Skipping this sheet.")
                    continue

                # Calculate correlation for each tumor marker and fluorescence feature
                for marker in tumor_markers:
                    for feature in fluorescence_features:
                        # Extract samples without missing values
                        valid_data = data.dropna(subset=[marker, feature])

                        if len(valid_data) < 3:  # At least 3 samples needed for correlation
                            continue

                        # Calculate Spearman correlation
                        r, p = stats.spearmanr(valid_data[marker], valid_data[feature])

                        # Determine significance level
                        sig_level = ""
                        if p < 0.001:
                            sig_level = "****"
                        elif p < 0.01:
                            sig_level = "***"
                        elif p < 0.05:
                            sig_level = "**"
                        elif p < p_threshold:  # Default is 0.1
                            sig_level = "*"

                        # Determine correlation strength
                        elif abs(r) >= 0.5:
                            strength = "Strong"
                        elif abs(r) >= 0.25:
                            strength = "Moderate"
                        else:
                            strength = "Weak"

                        # Add to results
                        all_results.append({
                            'Cancer Type': sheet_name.split(' ')[0],
                            'Tumor Marker': marker,
                            'Fluorescence Feature': feature,
                            'Correlation Coefficient': float(r),
                            'P-value': float(p),
                            'Significance': sig_level,
                            'Correlation Strength': strength,
                            'Direction': "Positive" if r > 0 else "Negative",
                            'Sample Size': len(valid_data)
                        })

            except Exception as e:
                print(f"Error processing sheet {sheet_name}: {str(e)}")
                continue

        # Check if any results exist
        if not all_results:
            raise ValueError("No valid correlation results were obtained from any sheet.")

        # Create summary DataFrame
        results_df = pd.DataFrame(all_results)

        # Create output directory
        output_path = Path(output_file)
        output_dir = output_path.parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        # Create a DataFrame with all correlations
        # Reshape data to create heatmap format: rows as fluorescence features, columns as tumor markers
        pivot_data = []

        for _, row in results_df.iterrows():
            cancer_type = row['Cancer Type']
            marker = row['Tumor Marker']
            feature = row['Fluorescence Feature']
            r = row['Correlation Coefficient']
            p = row['P-value']
            sig = row['Significance']

            # Create column identifier
            column_id = f"{cancer_type}_{marker}"

            pivot_data.append({
                'Feature': feature,
                'Column': column_id,
                'Correlation': r,
                'P-value': p,
                'Significance': sig
            })

        # Create pivot table
        pivot_df = pd.DataFrame(pivot_data)
        corr_matrix = pivot_df.pivot(index='Feature', columns='Column', values='Correlation')
        p_matrix = pivot_df.pivot(index='Feature', columns='Column', values='P-value')
        sig_matrix = pivot_df.pivot(index='Feature', columns='Column', values='Significance')

        # Create formatted string matrix with significance annotations
        formatted_corr = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
        is_significant = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns, data=False)

        for i in corr_matrix.index:
            for c in corr_matrix.columns:
                if pd.notnull(corr_matrix.loc[i, c]):
                    r_val = corr_matrix.loc[i, c]
                    sig = sig_matrix.loc[i, c] if pd.notnull(sig_matrix.loc[i, c]) else ""
                    formatted_corr.loc[i, c] = f"{r_val:.2f}{sig}"
                    is_significant.loc[i, c] = sig != ""
                else:
                    formatted_corr.loc[i, c] = ""

        # Save to Excel
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # Save summary results
            results_df.to_excel(writer, sheet_name='All Correlations', index=False)

            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['All Correlations']

            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'font_name': 'Arial',
                'font_size': 11,
                'align': 'center',
                'valign': 'vcenter'
            })

            cell_format = workbook.add_format({
                'font_name': 'Times New Roman',
                'font_size': 10,
                'align': 'center',
                'valign': 'vcenter'
            })

            # Apply formats
            for col_num, value in enumerate(results_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                worksheet.set_column(col_num, col_num, 15, cell_format)

            # Save correlation coefficient matrix
            corr_matrix.to_excel(writer, sheet_name='Correlation Matrix')
            p_matrix.to_excel(writer, sheet_name='P-value Matrix')

            # Format correlation coefficient matrix worksheet
            corr_sheet = writer.sheets['Correlation Matrix']
            p_sheet = writer.sheets['P-value Matrix']

            # Apply formats
            for sheet in [corr_sheet, p_sheet]:
                for col_num in range(len(corr_matrix.columns) + 1):
                    if col_num == 0:
                        sheet.write(0, col_num, "", header_format)
                    else:
                        sheet.write(0, col_num, corr_matrix.columns[col_num-1], header_format)
                        sheet.set_column(col_num, col_num, 12, cell_format)

                for row_num, value in enumerate(corr_matrix.index, 1):
                    sheet.write(row_num, 0, value, header_format)

        # Create heatmap
        # Custom color map: from RGB(90, 207, 249) to RGB(249, 132, 90)
        neg_color = np.array([90, 207, 249]) / 255  # Negative correlation color, blue
        pos_color = np.array([249, 132, 90]) / 255  # Positive correlation color, red
        mid_color = np.array([1, 1, 1])  # Middle color, white

        # Create custom color map
        colors = []
        for i in range(0, 51):
            # Gradient from blue to white
            ratio = i / 50
            color = (1 - ratio) * neg_color + ratio * mid_color
            colors.append(tuple(color))

        for i in range(0, 51):
            # Gradient from white to red
            ratio = i / 50
            color = (1 - ratio) * mid_color + ratio * pos_color
            colors.append(tuple(color))

        custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_diverging', colors, N=101)

        # Create rectangular heatmap
        plt.figure(figsize=(24, 14))

        # Set font
        plt.rcParams['font.family'] = 'Arial'

        # Mask NaN values
        mask = np.isnan(corr_matrix.values)

        # Create custom annotation function - Apply bold style to significant values
        def fmt_cell(val, is_sig):
            if val == "":
                return ""
            # If significant, use HTML style to increase font weight and size
            if is_sig:
                return f"$\\mathbf{{{val}}}$"  # Use LaTeX style for bold
            else:
                return val

        formatted_annotations = np.empty_like(formatted_corr.values, dtype=object)
        for i in range(formatted_corr.shape[0]):
            for j in range(formatted_corr.shape[1]):
                formatted_annotations[i, j] = fmt_cell(
                    formatted_corr.iloc[i, j],
                    is_significant.iloc[i, j]
                )

        # Draw heatmap
        ax = sns.heatmap(corr_matrix, annot=formatted_annotations, fmt="", cmap=custom_cmap,
                         vmin=-1, vmax=1, center=0, linewidths=1.0, mask=mask,
                         annot_kws={"size": 14, "ha": "center", "va": "center"})

        # Set title
        plt.title('CNT Features vs Tumor Markers Correlation Heatmap\n* p<0.1, ** p<0.05, *** p<0.01, **** p<0.001',
                  fontsize=20, pad=20, fontweight='bold')

        # Set label font and weight
        plt.ylabel('Fluorescence Features', fontsize=18, fontweight='bold')
        plt.xlabel('Tumor Markers by Cancer Type', fontsize=18, fontweight='bold')

        # Adjust tick labels
        plt.xticks(rotation=45, ha='right', fontsize=14, fontweight='bold')

        # Horizontal Y-axis labels
        plt.yticks(rotation=0, ha='right', fontsize=14, fontweight='bold')

        # Adjust heatmap cell shape to rectangular
        plt.gca().set_aspect(aspect='auto')  # Auto-adjust, not locked to square

        # Increase figure margins
        plt.tight_layout(pad=3.0)

        # Save image
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / "correlation_heatmap.png", dpi=dpi, bbox_inches='tight')
        plt.close()

        print(f"Analysis complete. Results saved to: {output_file}")
        print(f"Heatmap saved to: {plots_dir / 'correlation_heatmap.png'}")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()  # Print detailed error information


if __name__ == "__main__":
    # Configuration parameters
    config = {
        'input_file': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Source data for Mechanism analysis.xlsx'),  # Path to the input Excel file containing training data
        'output_file': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Results', 'Multicancer detection', '04_Mechanism analysis', 'spearman_correlation_results.xlsx'),  # Output file path for saving results
        'p_threshold': 0.1,  # Significance threshold
        'dpi': 600  # Image DPI
    }

    # Execute analysis
    analyze_tumor_markers_correlation(
        config['input_file'],
        config['output_file'],
        config['p_threshold'],
        config['dpi']
    )