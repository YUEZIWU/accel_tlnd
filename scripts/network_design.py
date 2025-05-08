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


--------------------------------------------------
Notes for future work:
Connect MST and MILP methods sequentially in the implementation. 
Once the MST algorithm has completed, the node clustering results could be stored.
In MILP algorithm, input the clusering results and bypass clustering and transformer placement steps.
--------------------------------------------------

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
    clusters = [{'nodes': [n], 'customers': [n[0]], 'center': (n[1], n[2])} for n in nodes]  # Cache initial centers
    
    improved = True
    while improved and len(clusters) > 1:
        improved = False
        
        # Use cached centers instead of recomputing
        centers = np.array([c['center'] for c in clusters])
        
        # Compute distances only for upper triangle to avoid redundant calculations
        cdists = distance_matrix(centers, centers)
        np.fill_diagonal(cdists, np.inf)
        
        # Get pairs efficiently
        rows, cols = np.where(cdists < np.inf)
        # Only keep upper triangle to avoid duplicates
        valid_pairs = rows < cols
        pairs = [(rows[i], cols[i], cdists[rows[i], cols[i]]) 
                for i in range(len(rows)) if valid_pairs[i]]
        pairs.sort(key=lambda x: x[2])  # Sort by distance

        # Try merging closest pairs first
        for (i, j, d) in pairs:
            if i >= len(clusters) or j >= len(clusters):
                continue
                
            new_nodes = clusters[i]['nodes'] + clusters[j]['nodes']
            if check_cluster_feasibility(new_nodes, max_dist):
                # Calculate center once and cache it
                new_center = compute_cluster_center(new_nodes)
                new_cluster = {
                    'nodes': new_nodes,
                    'customers': clusters[i]['customers'] + clusters[j]['customers'],
                    'center': new_center
                }
                
                # Remove old clusters and add new one
                c_del = sorted([i, j], reverse=True)
                for cidx in c_del:
                    del clusters[cidx]
                clusters.append(new_cluster)
                improved = True
                break
    # print(f"Found {len(clusters)} clusters")

    # Create final transformer locations and cluster assignments
    transformer_points = []
    customer_clusters = {}
    for i, c in enumerate(clusters):
        transformer_points.append(Point(*c['center']))  # Use cached center
        for customer_id in c['customers']:
            customer_clusters[customer_id] = i

    # Create GeoDataFrames
    transformer_gdf = gpd.GeoDataFrame(geometry=transformer_points, crs=nodes_gdf.crs)
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf['cluster'] = nodes_gdf.index.map(lambda x: customer_clusters[x])

    return nodes_gdf, transformer_gdf, clusters


# -------------------------------------------------
# 3. NETWORK DESIGN: MST or MILP
# -------------------------------------------------

# the minimum spanning tree (MST) function
def network_design_mst(nodes_gdf, transformer_gdf, clusters):

    # print("Creating LV networks using MST")
    # initialize variables
    transformer_points = []  # store Shapely Points for each cluster center
    lv_edges = []            # will store (cluster_id, start_id, end_id, x1, y1, x2, y2)
    customer_clusters = {}   # map from each customer node_id -> cluster_id
    n_nodes = len(nodes_gdf)  # number of real nodes

    for i,cluster in enumerate(clusters):
        # Track cluster assignments
        for customer_id in cluster['customers']:
            customer_clusters[customer_id] = i

        # get transformer locations
        tx_x, tx_y = compute_cluster_center(cluster['nodes'])
        transformer_points.append(Point(tx_x, tx_y))
        transformer_id = n_nodes + i  # Unique ID for transformer

        # Prepare cluster node information
        cluster_coords = np.array([[n[1], n[2]] for n in cluster['nodes']])
        cluster_node_ids = [n[0] for n in cluster['nodes']]
        # Combine transformer and cluster nodes
        points_with_tx = np.vstack([[tx_x, tx_y], cluster_coords])
        ids_with_tx = [transformer_id] + cluster_node_ids
        # Create MST for this cluster
        cluster_edges = create_mst(points_with_tx, point_ids=ids_with_tx)
        lv_edges.extend((i,) + edge for edge in cluster_edges)

    # Create GeoDataFrame for LV lines
    lv_gdf = gpd.GeoDataFrame(
        {
            'cluster_id': [edge[0] for edge in lv_edges],
            'start_id': [edge[1] for edge in lv_edges],
            'end_id': [edge[2] for edge in lv_edges],
            'start_type': ['transformer' if edge[1] >= n_nodes else 'customer' for edge in lv_edges],
            'end_type': ['transformer' if edge[2] >= n_nodes else 'customer' for edge in lv_edges],
        },
        geometry=[LineString([(edge[3], edge[4]), (edge[5], edge[6])]) for edge in lv_edges],
        crs=nodes_gdf.crs
    )
    lv_gdf['length'] = lv_gdf.geometry.length

    # Create MV network connecting tx
    transformer_coords = np.array([[p.x, p.y] for p in transformer_points])
    transformer_ids = [n_nodes + i for i in range(len(transformer_points))]
    mv_edges = create_mst(transformer_coords, point_ids=transformer_ids)
    
    # Create GeoDataFrame for MV lines
    mv_gdf = gpd.GeoDataFrame(
        {
            'start_id': [edge[0] for edge in mv_edges],
            'end_id': [edge[1] for edge in mv_edges],
            'start_tx': [edge[0] - n_nodes for edge in mv_edges],  # transformer index
            'end_tx': [edge[1] - n_nodes for edge in mv_edges],    # transformer index
        },
        geometry=[LineString([(edge[2], edge[3]), (edge[4], edge[5])]) for edge in mv_edges],
        crs=nodes_gdf.crs
    )
    mv_gdf['length'] = mv_gdf.geometry.length

    # Create GeoDataFrame for transformers
    transformer_gdf = gpd.GeoDataFrame(
        {
            'transformer_id': transformer_ids,
            'cluster_id': range(len(transformer_points)),
            'n_customers': [len(cluster['customers']) for cluster in clusters]
        },
        geometry=transformer_points,
        crs=nodes_gdf.crs
    )

    # ========== Update Customer Nodes ==========
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf['cluster'] = nodes_gdf.index.map(lambda x: customer_clusters[x])
    nodes_gdf['transformer_id'] = nodes_gdf['cluster'].map(lambda x: n_nodes + x)
    
    return nodes_gdf, transformer_gdf, lv_gdf, mv_gdf


# the mixed integer linear programming (MILP) function
def network_design_milp(nodes_gdf, transformer_gdf, clusters, max_dist, milp_params):

    # print("Creating LV networks using MILP")

    # Initialize variables
    n_nodes = len(nodes_gdf)
    transformer_points = []
    lv_edges = []
    customer_clusters = {}

    for i, cluster in enumerate(clusters):
        # Track cluster assignments
        for customer_id in cluster['customers']:
            customer_clusters[customer_id] = i

        # get transformer locations
        tx_x, tx_y = compute_cluster_center(cluster['nodes'])
        transformer_points.append(Point(tx_x, tx_y))
        transformer_id = n_nodes + i  # Unique ID for transformer
        
        # Prepare cluster node information
        cluster_coords = np.array([[n[1], n[2]] for n in cluster['nodes']])
        cluster_node_ids = [n[0] for n in cluster['nodes']]

        # combine transformer and cluster nodes
        points_with_tx = np.vstack([[tx_x, tx_y], cluster_coords])
        ids_with_tx = [transformer_id] + cluster_node_ids

        # solve MILP for this cluster
        feasible, c_lv_cost, c_lv_edges = milp_cmst_gurobi(points_with_tx, max_dist, milp_params, 
                                                           point_ids=ids_with_tx)
        
        if not feasible:
            print(f"Warning: Cluster {i} not feasible")
            continue
        
        # 6. Add cluster ID to edges and extend LV edges list
        lv_edges.extend((i,) + edge for edge in c_lv_edges)
   
    # Create LV Network GDF
    lv_gdf = gpd.GeoDataFrame(
        {
            'cluster_id': [edge[0] for edge in lv_edges],
            'start_id': [edge[1] for edge in lv_edges],
            'end_id': [edge[2] for edge in lv_edges],
            'start_type': ['transformer' if edge[1] >= n_nodes else 'customer' for edge in lv_edges],
            'end_type': ['transformer' if edge[2] >= n_nodes else 'customer' for edge in lv_edges],
        },
        geometry=[LineString([(edge[3], edge[4]), (edge[5], edge[6])]) for edge in lv_edges],
        crs=nodes_gdf.crs
    )
    lv_gdf['length'] = lv_gdf.geometry.length
    
    # create MV Network 
    # Create MST connecting all transformers
    transformer_coords = np.array([[p.x, p.y] for p in transformer_points])
    transformer_ids = [n_nodes + i for i in range(len(transformer_points))]
    mv_edges = create_mst(transformer_coords, point_ids=transformer_ids)
    
    mv_gdf = gpd.GeoDataFrame(
        {
            'start_id': [edge[0] for edge in mv_edges],
            'end_id': [edge[1] for edge in mv_edges],
            'start_tx': [edge[0] - n_nodes for edge in mv_edges],  # transformer index
            'end_tx': [edge[1] - n_nodes for edge in mv_edges],    # transformer index
        },
        geometry=[LineString([(edge[2], edge[3]), (edge[4], edge[5])]) for edge in mv_edges],
        crs=nodes_gdf.crs
    )
    mv_gdf['length'] = mv_gdf.geometry.length

    # create Transformer GDF
    transformer_gdf = gpd.GeoDataFrame(
        {
            'transformer_id': transformer_ids,
            'cluster_id': range(len(transformer_points)),
            'n_customers': [len(cluster['customers']) for cluster in clusters]
        },
        geometry=transformer_points,
        crs=nodes_gdf.crs
    )

    # ========== Update Customer Nodes ==========
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf['cluster'] = nodes_gdf.index.map(lambda x: customer_clusters[x])
    nodes_gdf['transformer_id'] = nodes_gdf['cluster'].map(lambda x: n_nodes + x)
    
    return nodes_gdf, transformer_gdf, lv_gdf, mv_gdf


# -------------------------------------------------
# 4. network design functions
# -------------------------------------------------
# this mst function will be used in LV mst layout if applicable and MV layout
def create_mst(points, point_ids=None):
    if point_ids is None:
        point_ids = list(range(len(points)))

    dist_matrix = distance_matrix(points, points)
    mst = minimum_spanning_tree(dist_matrix).toarray()

    edges = []
    n = len(points)
    for i in range(n):
        for j in range(n):
            if mst[i, j] > 0:
                x1, y1 = points[i]
                x2, y2 = points[j]
                start_id = point_ids[i]
                end_id   = point_ids[j]
                edges.append((start_id, end_id, x1, y1, x2, y2))
    return edges


def milp_cmst_gurobi(coords, max_dist, milp_params, point_ids=None):
    """
    Solve a CMST-like problem using Gurobi MILP:
    - coords: np.array of shape (n,2), coords[0] is transformer node ('root'), others are customers.
    """
    if point_ids is None:
        point_ids = list(range(len(coords)))

    n = len(coords)
    root = 0
    dist = np.sqrt((coords[:, 0, None] - coords[:, 0]) ** 2 + (coords[:, 1, None] - coords[:, 1]) ** 2)
    M = dist.max() * n + 1  # Big-M

    # Create a Gurobi model
    m = Model("CMST")
    m.setParam('OutputFlag', 0)  # No solver output, set to 1 for logs
    m.setParam('TimeLimit', milp_params["time_limit"])
    m.setParam('Threads', milp_params["threads"])

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

    # Extract solution edges with node IDs
    edges = []
    lv_cost = 0.0
    for i in range(n):
        for j in range(n):
            if x[(i, j)].X > 0.5:
                lv_cost += dist[i, j]
                edges.append((
                    point_ids[i],    # start node ID
                    point_ids[j],    # end node ID
                    coords[i][0],    # x1
                    coords[i][1],    # y1
                    coords[j][0],    # x2
                    coords[j][1]     # y2
                ))


    return True, lv_cost, edges
