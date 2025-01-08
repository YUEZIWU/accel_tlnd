import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from matplotlib_scalebar.scalebar import ScaleBar

# updated nodes_gdf
# LV, MV, transformers
# figures
# stats

def output_processing(nodes_gdf, transformer_gdf, lv_gdf, mv_gdf, config, elapsed_time):
    costs_params = config['costs_params']
    network_metrics = {
        'num_nodes': len(nodes_gdf),
        'num_transformers': len(transformer_gdf),
        'total_lv_length': sum(line.length for line in lv_gdf.geometry),
        'total_mv_length': sum(line.length for line in mv_gdf.geometry)
    }

    total_costs = network_metrics['num_transformers'] * costs_params['transformer_cost'] + \
                  network_metrics['total_lv_length'] * costs_params['lv_cost_per_meter'] + \
                  network_metrics['total_mv_length'] * costs_params['mv_cost_per_meter']
    costs_per_node = total_costs / network_metrics['num_nodes']
    lv_m_per_node = network_metrics['total_lv_length'] / network_metrics['num_nodes']
    mv_m_per_node = network_metrics['total_mv_length'] / network_metrics['num_nodes']
    node_per_transformer = network_metrics['num_nodes'] / network_metrics['num_transformers']

    output_metrics = {
        'Number of Nodes': network_metrics['num_nodes'],
        'Number of Transformers': network_metrics['num_transformers'],
        'Total LV Length (m)': network_metrics['total_lv_length'],
        'Total MV Length (m)': network_metrics['total_mv_length'],
        'Total Costs ($)': total_costs,
        'Costs Per Node ($)': costs_per_node,
        'LV Length Per Node (m)': lv_m_per_node,
        'MV Length Per Node (m)': mv_m_per_node,
        'Nodes Per Transformer': node_per_transformer,
        'Elapsed Time (s)': elapsed_time,
    }
    output_file = f'../results/{config["project_name"]}/network_metrics.txt'
    with open(output_file, "w") as file:
        for name, value in output_metrics.items():
            file.write(f"{name}, {value:.2f}\n")

    nodes_gdf.to_file(f'../results/{config["project_name"]}/nodes.gpkg', driver='GPKG')
    transformer_gdf.to_file(f'../results/{config["project_name"]}/transformers.gpkg', driver='GPKG')
    lv_gdf.to_file(f'../results/{config["project_name"]}/lv_lines.gpkg', driver='GPKG')
    mv_gdf.to_file(f'../results/{config["project_name"]}/mv_lines.gpkg', driver='GPKG')

    return None


def visualize_network(nodes_gdf, transformers_gdf, lv_gdf, mv_gdf, config):
    nodes_gdf = nodes_gdf.to_crs(epsg=3857)
    lv_gdf = lv_gdf.to_crs(epsg=3857)
    mv_gdf = mv_gdf.to_crs(epsg=3857)
    transformer_gdf = transformers_gdf.to_crs(epsg=3857)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    lv_gdf.plot(ax=ax, color='tab:blue', linewidth=2, label="LV")
    mv_gdf.plot(ax=ax, color='tab:orange', linewidth=3, linestyle='-', label="MV")
    transformer_gdf.plot(ax=ax, marker='s', color='#006633', markersize=70, zorder=3, label="TX")
    nodes_gdf.plot(ax=ax, marker='o', color='tab:red', markersize=20, label="Nodes")

    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, alpha=0.7)
    scalebar = ScaleBar(1, location='lower right', scale_loc='bottom', units='m', length_fraction=0.25)
    ax.add_artist(scalebar)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.grid(True)
    plt.legend()

    plt.savefig(f'../results/{config["project_name"]}/network.png', dpi=300)
    return None