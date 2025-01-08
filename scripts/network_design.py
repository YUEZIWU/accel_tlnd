"""
--------------------------------------------------
NETWORK DESIGN MODULE
--------------------------------------------------
Contains functions for:
1) Clustering and merging nodes into feasible groups.
2) Designing LV (low-voltage) and MV (medium-voltage) networks.
3) Optionally using Gurobi-based MILP for a Constrained MST approach.
--------------------------------------------------

2025.1.8, yuezi
"""


import numpy as np
import geopandas as gpd
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry import LineString, Point
from gurobipy import Model, GRB, quicksum


# -------------------------------------------------
# short functions
# -------------------------------------------------

# compute geometric center of a list of nodes
def compute_cluster_center(nodes):
    coords = np.array([[n[1], n[2]] for n in nodes])
    return coords[:, 0].mean(), coords[:, 1].mean()


# check if all nodes lie within max_dist from the cluster center
def check_cluster_feasibility(nodes, max_dist):
    cx, cy = compute_cluster_center(nodes)
    coords = np.array([[n[1], n[2]] for n in nodes])
    distances = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
    return np.all(distances <= max_dist)


# -------------------------------------------------
# 2. CLUSTER MERGING
# -------------------------------------------------
def merge_and_cluster(nodes_gdf, max_dist):
    coords = np.array([[geom.x, geom.y] for geom in nodes_gdf.geometry])
    nodes = [(i, coords[i][0], coords[i][1]) for i in range(len(coords))]
    clusters = [{'nodes': [n], 'customers': [n[0]]} for n in nodes]

    improved = True
    while improved and len(clusters) > 1:
        improved = False

        for c in clusters:
            c['center'] = compute_cluster_center(c['nodes'])
        centers = np.array([c['center'] for c in clusters])

        # Compute distances between cluster centers
        cdists = distance_matrix(centers, centers)
        np.fill_diagonal(cdists, np.inf)

        # Sort possible pairs by distance
        pairs = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                pairs.append((i, j, cdists[i, j]))
        pairs.sort(key=lambda x: x[2])  # sort the pairs by cdists

        # Try merging closest pairs first
        for (i, j, d) in pairs:
            if i >= len(clusters) or j >= len(clusters):
                continue
            new_nodes = clusters[i]['nodes'] + clusters[j]['nodes']

            # Check if merged cluster is feasible
            if check_cluster_feasibility(new_nodes, max_dist):
                new_cluster = {
                    'nodes': new_nodes,
                    'customers': clusters[i]['customers'] + clusters[j]['customers']
                }

                # Remove old clusters and add new one
                c_del = sorted([i, j], reverse=True)
                for cidx in c_del:
                    del clusters[cidx]
                clusters.append(new_cluster)
                improved = True
                break

    print(f"Found {len(clusters)} clusters")

    # Create final transformer locations and cluster assignments
    transformer_points = []
    customer_clusters = {}
    for i, c in enumerate(clusters):
        cx, cy = compute_cluster_center(c['nodes'])
        transformer_points.append(Point(cx, cy))
        for customer_id in c['customers']:
            customer_clusters[customer_id] = i

    # Create GeoDataFrames
    transformer_gdf = gpd.GeoDataFrame(geometry=transformer_points, crs=nodes_gdf.crs)

    # Add cluster assignments to nodes
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf['cluster'] = nodes_gdf.index.map(lambda x: customer_clusters[x])

    return nodes_gdf, transformer_gdf, clusters


# -------------------------------------------------
# 3. NETWORK DESIGN: MST or MILP
# -------------------------------------------------

# the minimum spanning tree (MST) function
def network_design_mst(nodes_gdf, transformer_gdf, clusters):

    print("Creating LV networks using MST")
    # Create LV networks for each cluster
    lv_edges = []

    for i,c in enumerate(clusters):
        # get transformer locations
        tx_x, tx_y = compute_cluster_center(c['nodes'])
        tx_coords = np.array([[tx_x, tx_y]])

        cluster_points = np.array([[n[1], n[2]] for n in c['nodes']])
        cluster_points_with_tx = np.vstack((tx_coords, cluster_points))

        # Create MST for cluster
        cluster_edges = create_mst(cluster_points_with_tx)
        lv_edges.extend(cluster_edges)

    # Create MV network connecting tx
    transformer_coords = np.array([[p.x, p.y] for p in transformer_gdf.geometry])
    mv_edges = create_mst(transformer_coords)

    # create gpds for lv and mv lines
    lv_lines = [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in lv_edges]
    lv_gdf = gpd.GeoDataFrame(geometry=lv_lines, crs=nodes_gdf.crs)
    mv_lines = [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in mv_edges]
    mv_gdf = gpd.GeoDataFrame(geometry=mv_lines, crs=nodes_gdf.crs)

    return nodes_gdf, transformer_gdf, lv_gdf, mv_gdf


# the mixed integer linear programming (MILP) function
def network_design_milp(nodes_gdf, transformer_gdf, clusters, max_dist, milp_params):

    print("Creating LV networks using MILP")
    lv_edges = []

    for i, c in enumerate(clusters):
        tx_x, tx_y = compute_cluster_center(c['nodes'])
        tx_coords = np.array([[tx_x, tx_y]])
        cluster_points = np.array([[n[1], n[2]] for n in c['nodes']])
        cluster_points_with_tx = np.vstack((tx_coords, cluster_points))

        # Solve the local LV design with MILP formulated in Gurobi
        feasible, c_lv_cost, c_lv_edges = milp_cmst_gurobi(cluster_points_with_tx, max_dist, milp_params)
        if not feasible:
            print(f"Cluster {i} not feasible")
        else:
            cluster_edges = c_lv_edges

        lv_edges.extend(cluster_edges)

        # MV network connecting tx
        transformer_coords = np.array([[p.x, p.y] for p in transformer_gdf.geometry])
        mv_edges = create_mst(transformer_coords)

        # create gpds for lv and mv lines
        lv_lines = [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in lv_edges]
        lv_gdf = gpd.GeoDataFrame(geometry=lv_lines, crs=nodes_gdf.crs)
        mv_lines = [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in mv_edges]
        mv_gdf = gpd.GeoDataFrame(geometry=mv_lines, crs=nodes_gdf.crs)

    return nodes_gdf, transformer_gdf, lv_gdf, mv_gdf


# -------------------------------------------------
# 4. network design functions
# -------------------------------------------------
# this mst function will be used in LV mst layout if applicable and MV layout
def create_mst(points):
    # Create a Minimum Spanning Tree (MST) from a set of points using SciPy.
    dist_matrix = distance_matrix(points, points)
    mst = minimum_spanning_tree(dist_matrix).toarray()
    edges = []
    for i in range(len(points)):
        for j in range(len(points)):
            if mst[i, j] > 0:
                edges.append((points[i][0], points[i][1], points[j][0], points[j][1]))
    return edges


def milp_cmst_gurobi(coords, max_dist, milp_params):
    """
    Solve a CMST-like problem using Gurobi MILP:
    - coords: np.array of shape (n,2), coords[0] is transformer node ('root'), others are customers.
    """

    n = len(coords)
    root = 0
    dist = np.sqrt((coords[:, 0, None] - coords[:, 0]) ** 2 + (coords[:, 1, None] - coords[:, 1]) ** 2)

    M = dist.max() * n + 1  # Big-M

    # Create a Gurobi model
    m = Model("CMST")
    m.setParam('OutputFlag', 0)  # No solver output, set to 1 for logs
    m.setParam('TimeLimit', milp_params["time_limit"])

    # Variables
    x = {}
    for i in range(n):
        for j in range(n):
            x[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    # d_vars[i] tracks the distance from the root to node i in the solution
    d_vars = {i: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"d_{i}") for i in range(n)}

    m.update()

    # Constraints
    # 1. Exactly one incoming edge for each node except root
    for j in range(1, n):
        m.addConstr(quicksum(x[(i, j)] for i in range(n) if i != j) == 1, name=f"incoming_{j}")

    # Root has no incoming edges
    m.addConstr(quicksum(x[(i, root)] for i in range(n) if i != root) == 0, name="incoming_root")

    # 2. Distance definitions
    m.addConstr(d_vars[root] == 0.0, name="dist_root")
    for i in range(n):
        for j in range(n):
            if i != j:
                m.addConstr(d_vars[j] >= d_vars[i] + dist[i, j] - M * (1 - x[(i, j)]),
                            name=f"dist_{i}_{j}")

    # 3. Distance limit
    for i in range(n):
        m.addConstr(d_vars[i] <= max_dist, name=f"maxDist_{i}")

    # Objective: minimize total LV length
    obj_expr = quicksum(dist[i, j] * x[(i, j)] for i in range(n) for j in range(n))
    m.setObjective(obj_expr, GRB.MINIMIZE)

    # Solve with time limit
    m.optimize()

    if m.Status not in [GRB.OPTIMAL, GRB.INTERRUPTED, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        return False, None, None

    # Check feasibility by distances
    dist_sol = [d_vars[i].X for i in range(n)]
    if any(dist_sol[i] > max_dist + 1e-9 for i in range(n)):
        return False, None, None

    # Extract solution edges
    edges = []
    lv_cost = 0.0
    for i in range(n):
        for j in range(n):
            if x[(i, j)].X > 0.5:
                lv_cost += dist[i, j]
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                edges.append((x1, y1, x2, y2))

    return True, lv_cost, edges

