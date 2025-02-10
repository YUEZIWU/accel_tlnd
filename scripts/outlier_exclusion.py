import numpy as np
from sklearn.cluster import DBSCAN

def exclude_outliers(nodes_gdf, eps_meters, min_samples):
    print("Excluding outliers")
    gdf_with_clusters = detect_spatial_outliers(nodes_gdf, eps_meters, min_samples)
    stats = analyze_clusters(gdf_with_clusters)
    # sub_group the included nodes
    nodes_gdf_incl = nodes_gdf[nodes_gdf['is_outlier'] == False].copy()
    nodes_gdf_incl = nodes_gdf_incl.reset_index(drop=True)
    nodes_gdf_excl = nodes_gdf[nodes_gdf['is_outlier'] == True].copy()
    nodes_gdf_excl = nodes_gdf_excl.reset_index(drop=True)

    return nodes_gdf_incl, nodes_gdf_excl, stats

def detect_spatial_outliers(nodes_gdf, eps_meters, min_samples):
    coords = np.vstack((nodes_gdf.geometry.x, nodes_gdf.geometry.y)).T
    db = DBSCAN(eps=eps_meters, min_samples=min_samples, metric='euclidean')
    clusters = db.fit_predict(coords)

    nodes_gdf['cluster'] = clusters
    nodes_gdf['is_outlier'] = nodes_gdf['cluster'] == -1

    return nodes_gdf


def analyze_clusters(gdf):
    total_points = len(gdf)
    outliers = sum(gdf['is_outlier'])
    clusters = len(set(gdf[gdf['cluster'] != -1]['cluster']))
    return {
        'total_points': total_points,
        'outliers': outliers,
        'clusters': clusters,
        'outlier_percentage': outliers / total_points * 100
    }