import os

def extract_experiment_data(base_folder):
    # Define the subfolders and subsubfolders structure
    subfolders = ["P97", "P95"]
    subsubfolders = ["RAND", "BC", "CC", "DC", "EC", "BC_CC", "BC_DC", "BC_EC", 
                     "CC_DC", "CC_EC", "DC_EC", "BC_CC_DC", "BC_CC_EC", "BC_DC_EC", 
                     "CC_DC_EC", "BC_CC_DC_EC"]

    # Initialize a dictionary to hold all the extracted data
    experiment_data = {}

    for subfolder in subfolders:
        for subsubfolder in subsubfolders:
            # Construct the path to the experiment_log.txt file
            file_path = os.path.join(base_folder, subfolder, subsubfolder, "experiment_log.txt")
            
            # Check if the file exists
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    if len(lines) >= 3:
                        # Extract the third line
                        third_line = lines[2]
                        # Split the line by commas to get the variables
                        acc, sens, spec, f1, auc = third_line.strip().split(',')
                        # Store the extracted data in the dictionary
                        experiment_data[(subfolder, subsubfolder)] = {
                            "acc": acc,
                            "sens": sens,
                            "spec": spec,
                            "f1": f1,
                            "auc": auc
                        }
            else:
                print(f"File not found: {file_path}")

    return experiment_data
    
results = {
    "zscore": extract_experiment_data("results/ZACPSGCN_CiteSeer_Table"),
    "minmax": extract_experiment_data("results/ACPSGCN_CiteSeer_Table")
}

# Define the row fields
rows = [
    "P95/RAND", "P95/BC", "P95/CC", "P95/DC", "P95/EC", "P95/BC_CC", "P95/BC_DC", "P95/BC_EC", 
    "P95/CC_DC", "P95/CC_EC", "P95/DC_EC", "P95/BC_CC_DC", "P95/BC_CC_EC", "P95/BC_DC_EC", 
    "P95/CC_DC_EC", "P95/BC_CC_DC_EC", 
    "P97/RAND", "P97/BC", "P97/CC", "P97/DC", "P97/EC", 
    "P97/BC_CC", "P97/BC_DC", "P97/BC_EC", "P97/CC_DC", "P97/CC_EC", "P97/DC_EC", "P97/BC_CC_DC", 
    "P97/BC_CC_EC", "P97/BC_DC_EC", "P97/CC_DC_EC", "P97/BC_CC_DC_EC"
]

# Open the results file for writing
with open("results/ACPSGCN_CiteSeer_results.txt", "w") as output_file:
    # Write the header
    output_file.write("minmax_acc,zscore_acc,minmax_sens,zscore_sens,minmax_spec,zscore_spec,minmax_f1,zscore_f1,minmax_auc,zscore_auc\n")
    
    for row in rows:
        subfolder, subsubfolder = row.split('/')
        # Get the data from both zscore and minmax
        minmax_data = results["minmax"].get((subfolder, subsubfolder), {"acc": "", "sens": "", "spec": "", "f1": "", "auc": ""})
        zscore_data = results["zscore"].get((subfolder, subsubfolder), {"acc": "", "sens": "", "spec": "", "f1": "", "auc": ""})
        
        # Write the row of data
        output_file.write(f"{minmax_data['acc']},{zscore_data['acc']},{minmax_data['sens']},{zscore_data['sens']},{minmax_data['spec']},{zscore_data['spec']},{minmax_data['f1']},{zscore_data['f1']},{minmax_data['auc']},{zscore_data['auc']}\n")
