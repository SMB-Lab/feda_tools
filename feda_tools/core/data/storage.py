import os

def save_results(output_directory, file_path, bi4_bur_df, bg4_df):
    bur_filename = os.path.splitext(os.path.basename(str(file_path)))[0]
    bur_filepath = os.path.join(output_directory, bur_filename) + ".bur"
    bi4_bur_df.to_csv(bur_filepath, sep='\t', index=False, float_format='%.6f')
    bg4_filepath = os.path.join(output_directory, bur_filename) + ".bg4"
    bg4_df.to_csv(bg4_filepath, sep='\t', index=False, float_format='%.6f')