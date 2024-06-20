import argparse
import os

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add an argument for the .sh file
parser.add_argument("sh_file", help="Path to the .sh file")

# Parse the command-line arguments
args = parser.parse_args()

to_comment = []
lines = []
# Open the .sh file
with open(args.sh_file, "r") as file:
    # Read each line in the file
    for i, line in enumerate(file):
        lines.append(line)
        # Check if the line contains "--dest"
        if "--dest" in line:
            # Split the line by "--dest"
            parts = line.split("--dest")
            # Get the argument that comes after "--dest"
            output_dir = parts[1].split(" ")[1].strip()

            # Check if "experiment_log.txt" file exists in each output_dir
            log_file = os.path.join(output_dir, "experiment_log.txt")
            if os.path.exists(log_file):
                to_comment.append(i)
                # Comment out the corresponding line from args.sh_file
                print(log_file)
    
with open(args.sh_file, "w") as file:
    for i, line in enumerate(lines):
        # print(line)
        if i in to_comment:
            file.write("# " + line)
        else:
            file.write(line)