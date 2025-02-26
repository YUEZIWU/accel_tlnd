# Structure Merging for Building Footprints

## Overview

When working with Google Open Buildings dataset, individual structures that belong to the same household or compound are often represented as separate points. This module implements an algorithm to identify and merge these related structures based on spatial proximity.

## Features

- **Spatial Proximity Merging**: Merges structures within a configurable radius (default: 20m)
- **Grid-Based Processing**: Efficiently handles large datasets by processing in spatial grid cells (default: 10km x 10km)
- **Parallel Computation**: Utilizes multi-core processing for faster execution
- **District-Level Processing**: Organizes processing by administrative boundaries
- **Area Aggregation**: Calculates total area of merged structures

## Usage

```bash
python merge_structures.py --input_dir "/path/to/input/directory" --output_dir "/path/to/output/directory" --radius 20 --grid_size 10000
```

## Parameters

- `--input_dir`: Directory containing district parquet files with building footprints
- `--output_dir`: Directory to save merged structure files
- `--radius`: Merging radius in meters (default: 20)
- `--grid_size`: Grid cell size in meters for processing (default: 10000)

### Input Format

The input directory should contain parquet files with building footprint data. Each file should include:
- `geometry`: Point geometry in UTM coordinates (e.g., EPSG:32636 for Uganda)
- `area_in_meters`: Area of the structure in square meters
- `d`: District or region identifier

### Output Format

The script generates parquet files with merged structures, including:
- `geometry`: Point geometry of the structure (original location)
- `district`: District or region identifier
- `origin_id`: Unique identifier for the structure
- `str_no`: Number of original structures merged into this point
- `AggArea_m2`: Total area of all merged structures in square meters

## Implementation Details

The merging algorithm works by:
1. Dividing the area into grid cells for efficient processing
2. Creating buffer zones around each structure
3. Identifying overlapping buffers to find nearby structures
4. Iteratively merging structures with overlapping buffers
5. Aggregating metadata (area, count) for the merged structures

## Dependencies

- geopandas
- pandas
- numpy
- joblib
- tqdm