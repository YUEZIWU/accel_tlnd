import os, yaml, time, pathlib
import geopandas as gpd
from outlier_exclusion import exclude_outliers
from network_design import merge_and_cluster, network_design_mst, network_design_milp
from write_outputs import output_processing, visualize_network

# Define directory paths
SCRIPT_DIR = pathlib.Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

# Edit the scenario parameters and config in the yaml file
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main(nodes_gdf, config):
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

    results_dir = PROJECT_DIR / "results" / config["project_name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    output_processing(nodes_gdf, transformer_gdf, lv_gdf, mv_gdf, excluded_nodes_gdf, 
                      config, elapsed_time, results_dir)
    visualize_network(nodes_gdf, transformer_gdf, lv_gdf, mv_gdf, excluded_nodes_gdf, 
                      config, results_dir)


if __name__ == "__main__":
    config = load_config(SCRIPT_DIR / "params.yaml")

    # any geospatial file format read by geopandas can be used
    nodes_gdf = gpd.read_file(str(PROJECT_DIR / config["my_shp"]))
    nodes_gdf.set_crs(epsg=config["epsg"], inplace=True)

    # make sure the crs of the nodes_gdf is set to the correct one.
    main(nodes_gdf, config)