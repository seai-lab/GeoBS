import os
import glob
import pandas as pd

def analyze_results(results_dir):
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return

    summary_data = []

    # Columns to calculate mean for
    metrics = ['true_class_prob', 'reciprocal_rank', 'hit@1', 'hit@3', 'gbs']

    print(f"Found {len(csv_files)} files. Processing...")

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            # Extract filename as model name
            filename = os.path.basename(file_path)
            model_name = filename.replace("eval_nabirds_ebird_meta_test_", "").replace("__trained50_debiased2.csv", "")
            
            # Calculate means
            stats = {'Model': model_name, 'Filename': filename}
            
            for metric in metrics:
                if metric in df.columns:
                    stats[metric] = df[metric].mean()
                else:
                    stats[metric] = None
            
            stats['count'] = len(df)
            summary_data.append(stats)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by model name for better readability
    if not summary_df.empty:
        summary_df = summary_df.sort_values('Model')
        
        # Reorder columns
        cols = ['Model'] + metrics + ['count', 'Filename']
        summary_df = summary_df[cols]

        print("\nSummary of Results:")
        print("-" * 100)
        # Set pandas display options to ensure all columns are shown
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.6f}'.format)
        
        print(summary_df.to_string(index=False))
        print("-" * 100)
        
        # Optional: Save to file
        output_file = "classification_results_summary.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"\nSummary saved to {output_file}")
    else:
        print("No valid data found.")

if __name__ == "__main__":
    target_dir = "TorchSpatial/eval_results/classification"
    if os.path.exists(target_dir):
        analyze_results(target_dir)
    else:
        print(f"Directory not found: {target_dir}")
