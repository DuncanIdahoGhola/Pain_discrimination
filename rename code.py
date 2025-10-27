import os
import glob

# Define the directory where your files are located
# If the script is in the same directory as the files, you can use '.'
directory = 'X' 

# Define the old and new subject identifiers
old_subject_id = 'sub-051'
new_subject_id = 'sub-052'

# Find all files that contain the old_subject_id in their name
# This will find both CSV and PNG files if they match the pattern
search_pattern = os.path.join(directory, f"*{old_subject_id}*")
files_to_rename = glob.glob(search_pattern)

print(f"Found {len(files_to_rename)} files to potentially rename.")

for old_filepath in files_to_rename:
    # Get the directory and the base filename
    dir_name, old_filename = os.path.split(old_filepath)
    
    # Create the new filename by replacing the old subject ID with the new one
    new_filename = old_filename.replace(old_subject_id, new_subject_id)
    
    # Construct the full new filepath
    new_filepath = os.path.join(dir_name, new_filename)
    
    try:
        os.rename(old_filepath, new_filepath)
        print(f"Renamed '{old_filename}' to '{new_filename}'")
    except OSError as e:
        print(f"Error renaming '{old_filename}': {e}")

print("Renaming process complete.")