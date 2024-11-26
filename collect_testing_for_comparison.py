import os
import shutil

# RUN ONLY AFTER RUNNING THE EXPERIMENTS

def collect_png_files(main_folder, destination_folder, pattern="_Rendering.png"):
    """
    Collects all PNG files matching the given pattern from subfolders
    and copies them to the destination folder, appending identifiers based on subfolder names to avoid duplicates.

    :param main_folder: Path to the main directory containing subfolders.
    :param destination_folder: Path to the folder where PNGs will be collected.
    :param pattern: Filename pattern to match. Defaults to '_Rendering.png'.
    """
    # Check if the main folder exists
    if not os.path.exists(main_folder):
        print(f"Error: The main folder '{main_folder}' does not exist.")
        return

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created destination folder: {destination_folder}")
    else:
        print(f"Destination folder already exists: {destination_folder}")

    # Initialize a counter for copied files
    copied_files = 0

    # Walk through all subdirectories
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith(pattern):
                source_path = os.path.join(root, file)

                # Get the subfolder name
                subfolder_name = os.path.basename(root)

                # Determine the identifier to append based on subfolder name
                identifier = ""
                if "PPO" in subfolder_name:
                    identifier = "_PPO"
                elif "TDQN" in subfolder_name:
                    identifier = "_TDQN"

                # Construct the new filename
                base, ext = os.path.splitext(file)
                if identifier:
                    new_filename = f"{base}{identifier}{ext}"
                else:
                    new_filename = file  # No identifier appended

                destination_path = os.path.join(destination_folder, new_filename)

                # Check if the new filename already exists to avoid overwriting
                if os.path.exists(destination_path):
                    print(f"Warning: {new_filename} already exists in the destination folder. Skipping copy.")
                    continue  # Skip copying this file

                # Copy the file
                try:
                    shutil.copy2(source_path, destination_path)  # copy2 preserves metadata
                    print(f"Copied: {source_path} to {destination_path}")
                    copied_files += 1
                except Exception as e:
                    print(f"Failed to copy {source_path}. Reason: {e}")

    print(f"\nTotal PNG files copied: {copied_files}")

if __name__ == "__main__":
    # ============================
    # Configure Your Paths Below
    # ============================

    # Option 1: Absolute Path (Recommended)
    # Replace the path below with the actual path to your main folder.
    # For Windows, use double backslashes '\\' or raw strings r"Path".
    main_folder = "Figs"  # Example for Windows
    # main_folder = "/path/to/your/MainFolder"    # Example for macOS/Linux

    # Define the destination folder inside the main folder
    destination_folder = os.path.join(main_folder, "CollectedPNGs")

    # Call the function to collect PNG files
    collect_png_files(main_folder, destination_folder)
    print("All desired PNG files have been collected.")
