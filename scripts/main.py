import os, yaml, time
import geopandas as gpd
from outlier_exclusion import exclude_outliers
from network_design import merge_and_cluster, network_design_mst, network_design_milp
from write_outputs import output_processing, visualize_network


# Edit the scenario parameters and config in the yaml file
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main(nodes_gdf, config):
    start_time = time.time()  # Start timer
    if config["outlier_exclusion_case"]:
        nodes_gdf, stats = exclude_outliers(nodes_gdf, config["eps_meters"], config["min_samples"])
    else:
        pass

    nodes_gdf, transformer_gdf, clusters = merge_and_cluster(nodes_gdf, config["max_dist"])

    if config["model_selection"] == 'mst':
        nodes_gdf, transformer_gdf, lv_gdf, mv_gdf = network_design_mst(nodes_gdf, transformer_gdf, clusters)
    elif config["model_selection"] == 'milp':
        nodes_gdf, transformer_gdf, lv_gdf, mv_gdf = network_design_milp(nodes_gdf, transformer_gdf, clusters,
                                                                         config["max_dist"], config["milp_params"])
    else:
        print("Model selection not recognized")

    elapsed_time = time.time() - start_time  # End timer and calculate elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    project_path = f'../results/{config["project_name"]}'
    if not os.path.exists(project_path):
        os.makedirs(project_path)
        print('mark 1')
    output_processing(nodes_gdf, transformer_gdf, lv_gdf, mv_gdf, config, elapsed_time)
    visualize_network(nodes_gdf, transformer_gdf, lv_gdf, mv_gdf, config)


if __name__ == "__main__":
    config = load_config("params.yaml")
    # any geospatial file format read by geopandas can be used
    nodes_gdf = gpd.read_file(config["my_shp"])
    nodes_gdf.set_crs(epsg=32636, inplace=True)
    # make sure the crs of the nodes_gdf is set to the correct one; for Uganda, we use EPSG:32636
    main(nodes_gdf, config)

