import os
import contextlib
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback

def get_extent(structures):
    """Get extent of structures with 10m buffer"""
    minx = min(structures.geometry.x) - 10
    maxx = max(structures.geometry.x) + 10
    miny = min(structures.geometry.y) - 10
    maxy = max(structures.geometry.y) + 10
    return minx, miny, maxx, maxy

def process_grid_cell(x_r, y_r, origin_points, split_range_m, merging_radius):
    """Process structures within a single grid cell"""
    # Filter points in current grid cell
    subset = origin_points[
        (origin_points.geometry.x >= x_r) & (origin_points.geometry.x < x_r + split_range_m) &
        (origin_points.geometry.y >= y_r) & (origin_points.geometry.y < y_r + split_range_m)
    ]
    if subset.empty:
        return gpd.GeoDataFrame()

    # Create buffers and spatial join
    points_buffer = subset.copy()
    points_buffer['buffers'] = points_buffer.geometry.buffer(merging_radius)
    points_buffer = points_buffer.explode('buffers', index_parts=True).set_geometry('buffers')
    del points_buffer['geometry']
    points_buffer.crs = origin_points.crs
    points_buffer_sj = gpd.sjoin(subset, points_buffer, how='inner')
    points_buffer_sj.index = range(len(points_buffer_sj))

    # Iteratively merge structures in overlapping buffers
    while True:
        buffer_counts = points_buffer_sj['origin_id_right'].value_counts()
        multi_str_list = buffer_counts[buffer_counts >= 2].index.tolist()
        if not multi_str_list:
            break
        
        buffer_id = multi_str_list[0]
        rows_in_buffer = points_buffer_sj[points_buffer_sj['origin_id_right'] == buffer_id]
        pts_id_list = rows_in_buffer['origin_id_left'].tolist()
        
        if pts_id_list:
            rows_of_pts = points_buffer_sj[points_buffer_sj['origin_id_left'].isin(pts_id_list)]
            add_row = rows_of_pts.loc[(rows_of_pts['origin_id_right'] == buffer_id) & 
                                    (rows_of_pts['origin_id_left'] == buffer_id)].copy()
            
            if not add_row.empty:
                add_row.loc[add_row.index[0], 'structure_no_left'] = len(rows_in_buffer)
                add_row.loc[add_row.index[0], 'aggr_area_m2'] = rows_in_buffer['area_in_meters_left'].sum()
                rows_to_drop = points_buffer_sj[
                    (points_buffer_sj['origin_id_left'].isin(pts_id_list)) |
                    (points_buffer_sj['origin_id_right'].isin(pts_id_list))
                ]
                points_buffer_sj.drop(rows_to_drop.index, inplace=True)
                points_buffer_sj = pd.concat([points_buffer_sj, add_row], ignore_index=True)
            else:
                points_buffer_sj.drop(rows_in_buffer.index, inplace=True)

    # Set area for non-merged points
    mask = points_buffer_sj.index.isin(points_buffer_sj[points_buffer_sj['structure_no_left'] == 1].index)
    points_buffer_sj.loc[mask, 'aggr_area_m2'] = points_buffer_sj.loc[mask, 'area_in_meters_left']

    return points_buffer_sj

def merge_structures(origin_points, district_name, merging_radius, split_range_m):
    """Merge nearby structures within a district"""
    origin_points['origin_id'] = range(len(origin_points))
    origin_points['structure_no'] = 1

    minx, miny, maxx, maxy = get_extent(origin_points)
    grid_coords = [(x, y) 
                   for x in np.arange(minx, maxx, split_range_m)
                   for y in np.arange(miny, maxy, split_range_m)]
    
    with parallel_backend('loky', n_jobs=-1):
        with tqdm(total=len(grid_coords), desc=f"Processing {district_name}") as pbar:
            with tqdm_joblib(pbar):
                results = Parallel()(
                    delayed(process_grid_cell)(x, y, origin_points, split_range_m, merging_radius)
                    for x, y in grid_coords
                )

    results = pd.concat(results, ignore_index=True)
    results = results.set_crs("EPSG:32636", allow_override=True)
    
    # Clean and rename columns
    results = results[['d_left', 'geometry', 'origin_id_left', 'structure_no_left', 'aggr_area_m2']]
    results.columns = ['district', 'geometry', 'origin_id', 'str_no', 'AggArea_m2']
    
    return results

def main(input_dir, output_dir, merging_radius, split_range_m):
    """Main function to process all district files"""
    os.makedirs(output_dir, exist_ok=True)
    
    district_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.parquet')])
    start_time = time.time()
    
    for district_file in district_files:
        points = gpd.read_parquet(os.path.join(input_dir, district_file))
        points = points[['area_in_meters', 'd', 'geometry']]
        
        results = merge_structures(points, district_name=district_file[:-8], 
                                 merging_radius=merging_radius, 
                                 split_range_m=split_range_m)
        
        # Add metadata
        results.attrs = {
            'crs': results.crs,
            'creation_date': pd.Timestamp.now().isoformat(),
            'source': 'Open Buildings v3',
            'district': district_file[:-8]
        }
        
        output_path = os.path.join(output_dir, f"{district_file[:-8]}_merged_structures.parquet")
        results.to_parquet(output_path, index=False, compression='snappy')
    
    print(f"Total processing time: {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge nearby structures in Uganda districts')
    parser.add_argument('--input_dir', required=True, help='Directory containing district parquet files')
    parser.add_argument('--output_dir', required=True, help='Directory to save merged structure files')
    # change this for different merging radius
    parser.add_argument('--radius', type=float, default=20, help='Merging radius in meters') 
    parser.add_argument('--grid_size', type=float, default=10000, help='Grid cell size in meters')
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.radius, args.grid_size) 