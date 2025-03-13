import os, yaml, time, pathlib
import geopandas as gpd
import pandas as pd
from outlier_exclusion import exclude_outliers
from network_design import merge_and_cluster, network_design_mst, network_design_milp
from write_outputs import output_processing, visualize_network
import multiprocessing
from functools import partial

# Define directory paths
SCRIPT_DIR = pathlib.Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

# Edit the scenario parameters and config in the yaml file
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main(nodes_gdf, config, file):
    start_time = time.time()  # Start timer
    if config["outlier_exclusion_case"]:
        nodes_gdf, excluded_nodes_gdf, outlier_stats = exclude_outliers(nodes_gdf, config["eps_meters"], config["min_samples"])
        print(f"Outliers: {outlier_stats['outliers']} ({outlier_stats['outlier_percentage']:.2f}%)")
    else:
        excluded_nodes_gdf = None
        outlier_stats = {'outliers': 0, 'outlier_percentage': 0}
        pass
    
    print("Clustering process is running ... ")
    nodes_gdf, transformer_gdf, clusters = merge_and_cluster(nodes_gdf, config["max_dist"])

    if config["model_selection"] == 'mst':
        print("MST method is running ... ")
        nodes_gdf, transformer_gdf, lv_gdf, mv_gdf = network_design_mst(nodes_gdf, transformer_gdf, clusters)
    elif config["model_selection"] == 'milp':
        print("MILP method is running ... ")
        nodes_gdf, transformer_gdf, lv_gdf, mv_gdf = network_design_milp(nodes_gdf, transformer_gdf, clusters,
                                                                         config["max_dist"], config["milp_params"])
    else:
        print("Model selection not recognized")

    elapsed_time = time.time() - start_time  # End timer and calculate elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Extract location and file name from the file path to create the results directory
    file_path = pathlib.Path(file)
    file_parts = str(file_path).split('/')
    
    mg_index = file_parts.index('mg_locations')
    location = file_parts[mg_index + 1]  # e.g., 'ABIM'
    file_name = file_parts[mg_index + 2].split('.')[0]  # e.g., 'ABIM_1'
    
    results_dir = PROJECT_DIR / "results" / "mg_pue_projects" / location / file_name
    results_dir.mkdir(parents=True, exist_ok=True)

    output_processing(nodes_gdf, transformer_gdf, lv_gdf, mv_gdf, excluded_nodes_gdf, 
                      config, elapsed_time, results_dir)
    visualize_network(nodes_gdf, transformer_gdf, lv_gdf, mv_gdf, excluded_nodes_gdf, 
                      config, results_dir)

def process_file(file, config):
    """Process a single file with the given configuration"""
    print(f"Processing file: {file}")
    try:
        nodes_gdf = gpd.read_file(file)
        main(nodes_gdf, config, file)
        return f"Successfully processed {file}"
    except Exception as e:
        return f"Error processing {file}: {str(e)}"
    

if __name__ == "__main__":

    files_to_run = pd.read_csv('/Users/wuyuezi/Desktop/all_cluster_files.csv')['file_path']
    config = load_config(SCRIPT_DIR / "params.yaml")
    # Extract and print the part of the path after /THESIS/
    
    files_to_process = files_to_run[:5]

    num_processes = multiprocessing.cpu_count() - 1 

    # print(files_to_process[0])
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Create a partial function with the config already set
        process_func = partial(process_file, config=config)
        # Map the function to the selected files and process in parallel
        results = list(pool.map(process_func, files_to_process))
    
    # Print results summary
    successful = sum(1 for r in results if r.startswith("Successfully"))
    print(f"Completed processing {len(results)} files ({successful} successful)")
        