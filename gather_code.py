import os
from pathlib import Path

def gather_code(directory, output_file, exclude_file=None):
    """
    Gather code from all files in directory and subdirectories.
    
    Args:
        directory: Root directory to start searching
        output_file: Path to output file
        exclude_file: Name of this script file to exclude
    """
    # Convert directory to Path object
    root_dir = Path(directory)
    
    # Open output file
    with open(output_file, 'w', encoding='utf-8') as out:
        # Walk through directory
        for path in root_dir.rglob('*'):
            # Skip directories and this script
            if path.is_dir() or path.name == exclude_file:
                continue
                
            try:
                # Read file content
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Write file path and content to output
                out.write(f"\n{'='*80}\n")
                out.write(f"File: {path}\n")
                out.write(f"{'='*80}\n\n")
                out.write(content)
                out.write("\n\n")
                
                print(f"Processed: {path}")
                
            except Exception as e:
                print(f"Error processing {path}: {e}")

def main():
    # Directory to search
    directory = "C:/aishrink"
    
    # Output file path
    output_file = "C:/aishrink/all_code.txt"
    
    # Name of this script
    this_script = "gather_code.py"
    
    print(f"Gathering code from {directory}...")
    gather_code(directory, output_file, this_script)
    print(f"\nDone! Output written to {output_file}")

if __name__ == "__main__":
    main() 