#!/usr/bin/env python
import os
import json
import argparse
import numpy as np


def rescale_camera_parameters(input_file, output_file, scale_factor):
    """
    Rescale depth values and camera positions in a camera_parameters.json file
    
    Args:
        input_file (str): Path to input camera_parameters.json
        output_file (str): Path to output camera_parameters.json
        scale_factor (float): Scale factor to apply to depth and camera positions
    """
    print(f"Reading camera parameters from: {input_file}")
    
    # Read the input file
    with open(input_file, 'r') as f:
        params = json.load(f)
    
    # Check if the file already has scaled values
    if 'depth_scale_factor' in params:
        original_scale = params['depth_scale_factor']
        print(f"Input file already has a scale factor of {original_scale}")
        print(f"Will apply additional scale factor of {scale_factor}")
        total_scale = original_scale * scale_factor
    else:
        print(f"Applying scale factor: {scale_factor}")
        total_scale = scale_factor
    
    # Determine units
    original_units = params.get('original_units', 'unknown')
    if scale_factor == 0.001:
        scaled_units = 'meters'
    elif scale_factor == 1000:
        scaled_units = 'millimeters'
    else:
        scaled_units = 'custom'
    
    # Scale center, extent, and radius
    if 'center' in params:
        params['center'] = [val * scale_factor for val in params['center']]
    if 'extent' in params:
        params['extent'] = [val * scale_factor for val in params['extent']]
    if 'radius' in params:
        params['radius'] = params['radius'] * scale_factor
    
    # Scale view data
    views_modified = 0
    for view in params.get('views', []):
        # Scale camera position
        if 'camera_position' in view:
            view['camera_position'] = [val * scale_factor for val in view['camera_position']]
        
        # Scale depth values
        if 'depth_min' in view:
            view['depth_min'] *= scale_factor
        if 'depth_max' in view:
            view['depth_max'] *= scale_factor
        if 'depth_scale' in view:
            view['depth_scale'] *= scale_factor
        
        views_modified += 1
    
    # Add metadata about scaling
    params['depth_scale_factor'] = total_scale
    params['original_units'] = original_units
    params['scaled_units'] = scaled_units
    
    # Write the output file
    with open(output_file, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"Modified {views_modified} views")
    print(f"Saved rescaled parameters to: {output_file}")
    print(f"Total scale factor: {total_scale} ({original_units} to {scaled_units})")


def main():
    parser = argparse.ArgumentParser(description='Rescale depth values in camera_parameters.json')
    parser.add_argument('input_file', type=str,
                        help='Path to input camera_parameters.json')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to output camera_parameters.json (default: overwrite input)')
    parser.add_argument('--scale', '-s', type=float, default=0.001,
                        help='Scale factor to apply (default: 0.001, mm to m)')
    parser.add_argument('--to-mm', action='store_true',
                        help='Convert to millimeters (equivalent to --scale 1000)')
    parser.add_argument('--to-m', action='store_true',
                        help='Convert to meters (equivalent to --scale 0.001)')
    
    args = parser.parse_args()
    
    # Handle unit conversion flags
    if args.to_mm:
        scale_factor = 1000
    elif args.to_m:
        scale_factor = 0.001
    else:
        scale_factor = args.scale
    
    # Set output file
    if args.output is None:
        # Create a backup of the original file
        base_dir = os.path.dirname(args.input_file)
        base_name = os.path.basename(args.input_file)
        backup_path = os.path.join(base_dir, f"{os.path.splitext(base_name)[0]}_original.json")
        
        import shutil
        shutil.copy2(args.input_file, backup_path)
        print(f"Created backup of original file: {backup_path}")
        
        output_file = args.input_file
    else:
        output_file = args.output
    
    try:
        rescale_camera_parameters(args.input_file, output_file, scale_factor)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 