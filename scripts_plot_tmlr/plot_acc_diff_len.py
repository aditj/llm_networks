import os
import json


import os
import json
import matplotlib.pyplot as plt

# You can pass one or multiple experiment names here
result_type = 'event'


exp_names = [
    result_type + "_len1",
    result_type + "_len50",
    result_type + "_len100"
]
exp_names1 = [
    result_type + "_len1",
    result_type + "_len50",
    result_type + "_len100"
]

flag = 'A'


save_name = "different_output_len_" + result_type + "_" + flag
# Base directories
base_root = "exps1_tmlr"
save_root = "exps1_tmlr"


plt.figure(figsize=(8,5))

for exp_name, label in zip(exp_names, exp_names1):
    base_path = os.path.join(base_root, exp_name)

    # Load data
    combined_data = []
    for i in range(100):
        folder_path = os.path.join(base_path, str(i))
        file_path = os.path.join(folder_path, "ans.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
                combined_data.append(data)

    print(f"{label}: runs={len(combined_data)}, lines={len(combined_data[0])}, timesteps={len(combined_data[0][0])}")

    # Count 'A' proportions
    num_timesteps = len(combined_data[0][0])
    a_props = [0] * num_timesteps

    
    total_case = 0
    for run in combined_data:           # 100 runs
        count_A = sum(1 for occ in run if occ[-1] == flag)
        
        # if "llama" in label:
        #     if count_A  < 1:
        #         continue  # ignore this run
        total_case += len(combined_data[0])
        for line in run:                # 100 models
            
            for t, val in enumerate(line):  # timesteps
                
                if val == flag:
                    a_props[t] += 1
    # for run in combined_data:           # 100 runs
    #     for line in run:                # 100 lines
    #         for t, val in enumerate(line):  # timesteps
    #             if val == 'A':
    #                 a_props[t] += 1

    a_props = [(count / total_case ) for count in a_props]  # normalize
    print(total_case)
    # Plot with label from exp_labels
    plt.plot(range(num_timesteps), a_props, marker='o', label=label)

# Labels
plt.xlabel("Timestep")
plt.ylabel("Accuracy Proportion (fraction of 'A')")
plt.title("Accuracy Proportion per Timestep")
plt.grid(True)
plt.legend()

# Save
save_path = os.path.join(save_root, save_name + ".png")
plt.savefig(save_path)
plt.close()

print("Saved figure to:", save_path)


# output_path
