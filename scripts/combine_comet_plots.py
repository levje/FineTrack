import json
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

sns.set_style("darkgrid")


def read_json_files(json_paths):
    """Reads and parses multiple JSON files."""
    data = []
    for path in json_paths:
        with open(path, 'r') as file:
            data.extend(json.load(file))
    return data

def plot_data(data, output_file=None):
    """Plots the data as is. Saves or shows the plot."""
    plt.figure(figsize=(10, 6))
    for item in data:
        plt.plot(item['x'], item['y'], label=item['name'])
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Combined Line Plot")
    plt.legend()
    plt.grid()

    if output_file:
        plt.savefig(output_file)
        print(f"Figure saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Combine and plot multiple JSON files.")
    parser.add_argument(
        "json_files", 
        nargs="+", 
        help="Paths to the JSON files to combine and plot."
    )
    parser.add_argument(
        "--output-file", 
        type=str, 
        default=None, 
        help="Path to save the plot. If not specified, the plot is displayed."
    )
    args = parser.parse_args()

    # Read and plot data
    data = read_json_files(args.json_files)
    plot_data(data, output_file=args.output_file)

if __name__ == "__main__":
    main()