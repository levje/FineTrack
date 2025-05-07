import json
import argparse

import numpy as np

parser = argparse.ArgumentParser(description="Average comet data from a single JSON file.")
parser.add_argument("json_file", help="Path to the JSON file containing comet data.")
parser.add_argument("average_key", help="Key to average the data under.")
parser.add_argument("output_file", help="Path to the output JSON file to save the averaged data.")
parser.add_argument("--max_x", type=float, help="Maximum x-value to consider.")

args = parser.parse_args()
json_file = args.json_file
average_key = args.average_key
output_file = args.output_file

with open(json_file, "r") as f:
    comet_data = json.load(f)

x_check = None
type_check = None

for experiment in comet_data:
    if x_check is None:
        x_check = experiment["x"]
        type_check = experiment["type"]
    else:
        if type_check != experiment["type"]:
            raise ValueError(f"type values do not match for experiment {experiment}")

# Iterate through each experiment and average the y-values
y_values = []
x_values = None
nb_values = np.inf
for experiment in comet_data:
    y = experiment["y"]
    # nb_values = min(nb_values, len(y))
    if len(y) < nb_values:
        nb_values = len(y)
        x_values = experiment["x"]

    y_values.append(y)

# Truncate y-values to the minimum length
for i in range(len(y_values)):
    y_values[i] = y_values[i][:nb_values]

if args.max_x is not None:
    # Filter x-values based on the max_x argument
    x_values = [x for x in x_values if x <= args.max_x]
    y_values = [y[:len(x_values)] for y in y_values]

averaged_data = {
    "x": x_values,
    "type": experiment["type"],
    "name": average_key
}

y_values = np.array(y_values)
averaged_data["std"] = np.std(y_values, axis=0).tolist()
averaged_data["y"] = np.mean(y_values, axis=0).tolist()

# Save the averaged data to a new JSON file
with open(output_file, "w") as f:
    json.dump([averaged_data], f, indent=4)    

print("Averaged data saved to", output_file)
