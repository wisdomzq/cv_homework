import os
import shutil
import glob

def organize_jaffe(source_dir, target_dir):
    """
    Organize JAFFE dataset into subdirectories based on expression codes.
    
    JAFFE filename format: KA.AN1.39.tiff
    Code is the second part: AN, DI, FE, HA, NE, SA, SU
    """
    
    # Expression code to name mapping
    expression_map = {
        'AN': 'Angry',
        'DI': 'Disgust',
        'FE': 'Fear',
        'HA': 'Happy',
        'NE': 'Neutral',
        'SA': 'Sad',
        'SU': 'Surprise'
    }
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory: {target_dir}")

    # Get all tiff files
    files = glob.glob(os.path.join(source_dir, "*.tiff"))
    print(f"Found {len(files)} images in {source_dir}")
    
    count = 0
    for file_path in files:
        filename = os.path.basename(file_path)
        # Parse filename: KA.AN1.39.tiff -> parts=['KA', 'AN1', '39', 'tiff']
        parts = filename.split('.')
        if len(parts) < 2:
            print(f"Skipping invalid filename: {filename}")
            continue
            
        # The expression code is the first 2 characters of the second part (e.g., 'AN1' -> 'AN')
        code_part = parts[1]
        code = code_part[:2]
        
        if code in expression_map:
            expression_name = expression_map[code]
            
            # Create class directory if it doesn't exist
            class_dir = os.path.join(target_dir, expression_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
                
            # Copy file
            dest_path = os.path.join(class_dir, filename)
            shutil.copy2(file_path, dest_path)
            count += 1
        else:
            print(f"Unknown expression code '{code}' in file {filename}")

    print(f"Successfully organized {count} images into '{target_dir}'")

if __name__ == "__main__":
    # Assuming the script is run from hw3/PCA
    # And the images are in hw3/PCA/jaffe/jaffe
    
    current_dir = os.getcwd()
    source_path = os.path.join(current_dir, "jaffe", "jaffe")
    target_path = os.path.join(current_dir, "jaffe_organized")
    
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    
    organize_jaffe(source_path, target_path)
