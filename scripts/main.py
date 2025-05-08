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
    else:
        excluded_nodes_gdf = None
        outlier_stats = {'outliers': 0, 'outlier_percentage': 0}
        pass
    
    # Cluster nodes and place transformers
    nodes_gdf, transformer_gdf, clusters = merge_and_cluster(nodes_gdf, config["max_dist"])

    # Design network using selected method
    if config["model_selection"] == 'mst':
        nodes_gdf, transformer_gdf, lv_gdf, mv_gdf = network_design_mst(nodes_gdf, transformer_gdf, clusters)
    elif config["model_selection"] == 'milp':
        nodes_gdf, transformer_gdf, lv_gdf, mv_gdf = network_design_milp(nodes_gdf, transformer_gdf, clusters,
                                                                         config["max_dist"], config["milp_params"])
    else:
        print("Model selection not recognized")

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    file_name = file.split('/')[-1].split('.')[0]

    results_dir = PROJECT_DIR / "results" / file_name
    results_dir.mkdir(parents=True, exist_ok=True)

    output_processing(nodes_gdf, transformer_gdf, lv_gdf, mv_gdf, excluded_nodes_gdf, config, elapsed_time, results_dir)
    # visualize_network(nodes_gdf, transformer_gdf, lv_gdf, mv_gdf, excluded_nodes_gdf, config, results_dir)

# Define processing function directly within the multiprocessing context
def process_worker(file, config):
    try:
        nodes_gdf = gpd.read_file(file)
        print(f"Processing {file} with {len(nodes_gdf)} nodes")
        main(nodes_gdf, config, file)
        return f"Successfully processed {file}"
    except Exception as e:
        print(f"Error in worker: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error processing {file}: {str(e)}"

if __name__ == "__main__":
    
    parallel_processing = False

    if parallel_processing:
        # files_to_run = pd.read_csv('inputs/cluster_for_design.csv')['path']
        
        config = load_config(SCRIPT_DIR / "params.yaml")
        
        data_from = "inputs"
        files_to_run = [f for f in os.listdir(data_from) if f.endswith('.geojson')]
        files_to_process = [os.path.join(data_from, file) for file in files_to_run]

        num_processes = multiprocessing.cpu_count() - 2
        
        worker_with_config = partial(process_worker, config=config)
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(pool.imap_unordered(worker_with_config, files_to_process))

        # Print summary
        successful = sum(1 for r in results if r.startswith("Successfully"))
        print(f"Completed processing {len(results)} files ({successful} successful)")
    
    else:
        # files_to_run = pd.read_csv('inputs/cluster_for_design_group_1_milp.csv')['path']
        config = load_config(SCRIPT_DIR / "params.yaml")
        
        data_from = ".../mg_locations"
        # files_to_process = [os.path.join(data_from, file) for file in files_to_run]

        files_to_process = ['.../nodes_selected.geojson']
        # Process files one by one
        for file in files_to_process:
            print(f"Processing {file}")
            try:
                nodes_gdf = gpd.read_file(file)
                main(nodes_gdf, config, file)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
        
        print(f"Completed processing {len(files_to_process)} files")
