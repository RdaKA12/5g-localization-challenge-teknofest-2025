#!/usr/bin/env python3.11
import tensorflow as tf
import sionna
from sionna.rt import Scene, RadioMaterial, load_scene, SceneObject, Transmitter, Receiver, PlanarArray, PathSolver, Paths
import pandas as pd
import numpy as np
import os
import pickle

def get_configured_scene(xml_file_path):
    print("Loading scene and assigning materials...")
    # Adjusted scattering coefficients based on typical values, ensure they are reasonable.
    # Concrete: Lower scattering. Vegetation: Higher scattering.
    py_concrete = RadioMaterial(name="py_concrete", relative_permittivity=6.0, conductivity=0.05, scattering_coefficient=0.2)
    py_ground_material = RadioMaterial(name="py_ground", relative_permittivity=4.5, conductivity=0.005, scattering_coefficient=0.1)
    py_vegetation_material = RadioMaterial(name="py_vegetation", relative_permittivity=10.0, conductivity=0.1, scattering_coefficient=0.8) # Increased for vegetation
    py_metal = RadioMaterial(name="py_metal", relative_permittivity=1.0, conductivity=1.0e7, scattering_coefficient=0.05) # Low for smooth metal

    scene = load_scene(xml_file_path)
    print(f"Scene loaded from {xml_file_path}.")

    # Assign materials to objects based on names in the XML/OBJ
    # This assumes object names in scene.xml like 'ground', 'building_xxx', 'vegetation_xxx', 'bs_xxx'
    default_material_assigned = False
    for obj_name, scene_obj in scene.objects.items():
        if not isinstance(scene_obj, SceneObject):
            continue
        obj_name_lower = obj_name.lower()
        assigned = False
        if "building" in obj_name_lower: # Catches building, Building, etc.
            scene_obj.radio_material = py_concrete
            assigned = True
        elif "ground" in obj_name_lower:
            scene_obj.radio_material = py_ground_material
            assigned = True
        elif "vegetation" in obj_name_lower or "tree" in obj_name_lower or "plant" in obj_name_lower:
            scene_obj.radio_material = py_vegetation_material
            assigned = True
        elif "bs_" in obj_name_lower or "basestation" in obj_name_lower or obj_name_lower.startswith("mesh") : # Assuming base_stations.obj meshes are named Mesh_0, Mesh_1 etc.
            scene_obj.radio_material = py_metal # Base stations often metallic
            assigned = True
        
        if assigned:
            print(f"Assigned material \'{scene_obj.radio_material.name}\' to object \'{obj_name}\'")
        else:
            print(f"Warning: No specific material rule for \'{obj_name}\'. It will use its default material from XML or a global default if any.")
            # Optionally assign a default material if none of the rules match
            # scene_obj.radio_material = py_concrete # Example default
            # default_material_assigned = True

    # if default_material_assigned:
    #     print("Assigned a default material to some objects.")
    print("Material assignment process completed.")
    return scene

def run_full_simulation_pathsolver_call():
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Sionna version: {sionna.__version__}")

    # Paths
    xml_file_path = "/home/ubuntu/upload/scene.xml"
    hucre_excel_path = "/home/ubuntu/upload/hucre_bilgileri.xlsx"
    obj_centroids_path = "/home/ubuntu/base_station_obj_centroids.csv"
    dl_params_excel_path = "/home/ubuntu/upload/5g_dl.xlsx"
    output_dir = "/home/ubuntu/simulation_results"
    os.makedirs(output_dir, exist_ok=True)
    paths_output_file = os.path.join(output_dir, "propagation_paths.pkl")
    cir_output_file = os.path.join(output_dir, "channel_impulse_responses.pkl")
    simulation_run_summary_file = os.path.join(output_dir, "simulation_run_summary.txt")

    # Load scene with materials
    scene = get_configured_scene(xml_file_path)

    # Carrier frequency
    carrier_frequency = 3.5e9 # Default
    try:
        dl_df = pd.read_excel(dl_params_excel_path)
        if 'CarrierFrequency_GHz' in dl_df.columns:
            carrier_frequency = dl_df['CarrierFrequency_GHz'].iloc[0] * 1e9
        print(f"Carrier frequency set to: {carrier_frequency/1e9} GHz")
    except Exception as e:
        print(f"Error reading carrier frequency from {dl_params_excel_path}: {e}. Using default {carrier_frequency/1e9} GHz.")
    scene.frequency = np.float32(carrier_frequency)

    # Define antenna arrays for the scene (these are global for all TX/RX if not overridden)
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
    print("Scene tx_array and rx_array set to isotropic PlanarArray.")

    # Load base station information
    try:
        bs_df_excel = pd.read_excel(hucre_excel_path)
        if len(bs_df_excel) > 9: bs_df_excel = bs_df_excel.head(9)
        print(f"Loaded {len(bs_df_excel)} base station records from {hucre_excel_path}")
        
        bs_df_obj_centroids = pd.read_csv(obj_centroids_path)
        print(f"Loaded {len(bs_df_obj_centroids)} base station OBJ centroids from {obj_centroids_path}")

        if len(bs_df_excel) != len(bs_df_obj_centroids):
            print(f"Warning: Mismatch in number of records. Excel: {len(bs_df_excel)}, OBJ Centroids: {len(bs_df_obj_centroids)}. Using minimum of {min(len(bs_df_excel), len(bs_df_obj_centroids))}.")
            num_bs = min(len(bs_df_excel), len(bs_df_obj_centroids))
            bs_df_excel = bs_df_excel.head(num_bs)
            bs_df_obj_centroids = bs_df_obj_centroids.head(num_bs)
        else:
            num_bs = len(bs_df_excel)

    except Exception as e:
        print(f"Error loading base station data: {e}. Aborting simulation.")
        return

    # Create Transmitters
    if not all(col in bs_df_excel.columns for col in ['Height', 'Azimuth']):
        print(f"Error: Missing 'Height' or 'Azimuth' in {hucre_excel_path}. Aborting simulation.")
        return
    if not all(col in bs_df_obj_centroids.columns for col in ['X_obj', 'Y_obj']):
        print(f"Error: Missing 'X_obj' or 'Y_obj' in {obj_centroids_path}. Aborting simulation.")
        return

    for i in range(num_bs):
        excel_row = bs_df_excel.iloc[i]
        obj_row = bs_df_obj_centroids.iloc[i]
        
        tx_name = f"bs_{excel_row.get('Cell_ID', obj_row.get('MeshName', i+1))}"
        tx_position = np.array([float(obj_row['X_obj']), float(obj_row['Y_obj']), float(excel_row['Height'])], dtype=np.float32)
        alpha_yaw_deg = float(excel_row['Azimuth'])
        alpha = np.deg2rad(alpha_yaw_deg)
        beta = np.deg2rad(0.0) # Assuming 0 tilt (beta) and 0 roll (gamma)
        gamma = np.deg2rad(0.0)
        tx_orientation = np.array([alpha, beta, gamma], dtype=np.float32)
        
        tx = Transmitter(name=tx_name, position=tx_position, orientation=tx_orientation)
        scene.add(tx)
        print(f"Added Transmitter: {tx_name} at {tx_position} with orientation (radians) {tx_orientation} (Yaw_deg: {alpha_yaw_deg})")

    print(f"Total {len(scene.transmitters)} transmitters added to the scene.")

    # Create Receivers Grid
    grid_min_x, grid_max_x, num_x = -10, 10, 3 # Define grid boundaries and density - RESTORED SMALL SCALE FOR FINAL OUTPUTS
    grid_min_y, grid_max_y, num_y = -10, 10, 3 # RESTORED SMALL SCALE FOR FINAL OUTPUTS
    rx_height = 1.5 # Standard UE height
    rx_positions = []
    for x_val in np.linspace(grid_min_x, grid_max_x, num_x):
        for y_val in np.linspace(grid_min_y, grid_max_y, num_y):
            rx_positions.append([x_val, y_val, rx_height])
    for i, pos in enumerate(rx_positions):
        rx_name = f"rx_grid_{i}"
        rx_pos_np = np.array(pos, dtype=np.float32)
        rx_orientation_np = np.array([0,0,0], dtype=np.float32) # Default orientation for UEs
        rx = Receiver(name=rx_name, position=rx_pos_np, orientation=rx_orientation_np)
        scene.add(rx)
    print(f"Total {len(scene.receivers)} receivers added to the scene in a grid.")

    # Scene configuration for path computation
    scene.synthetic_array = True 
    print(f"Scene synthetic_array set to: {scene.synthetic_array}")

    # Instantiate a path solver
    p_solver = PathSolver()
    print("PathSolver instantiated.")

    # Path computation parameters
    path_comp_max_depth = 5 # REDUCED FOR TEST
    path_comp_num_samples_scattering = tf.constant(1_000_000, dtype=tf.int32) # For diffuse reflection
    path_comp_los = True
    path_comp_specular_reflection = True
    path_comp_diffuse_reflection = True  # Enable scattering via diffuse reflection
    path_comp_refraction = True          # Enable transmission/refraction
    path_comp_diffraction = True         # Attempt to enable diffraction
    path_comp_synthetic_array_call = False # As per tutorial for p_solver() call, may differ from scene.synthetic_array
    path_comp_seed = 42                  # For reproducibility

    print(f"Path computation parameters for p_solver(): max_depth={path_comp_max_depth}, num_samples_scattering={path_comp_num_samples_scattering.numpy()}, LoS={path_comp_los}, SpecularReflection={path_comp_specular_reflection}, DiffuseReflection={path_comp_diffuse_reflection}, Refraction={path_comp_refraction}, Diffraction={path_comp_diffraction}, SyntheticArrayCall={path_comp_synthetic_array_call}, Seed={path_comp_seed}")

    # Run Simulation (Compute Paths using PathSolver instance call)
    print("Starting path computation using p_solver(scene=scene, ...)... This may take some time.")
    try:
        paths = p_solver(scene=scene,
                         max_depth=path_comp_max_depth,
                         los=path_comp_los,
                         specular_reflection=path_comp_specular_reflection,
                         diffuse_reflection=path_comp_diffuse_reflection,
                         refraction=path_comp_refraction,
                         # diffraction=path_comp_diffraction, # Removed as it caused TypeError
                         # num_samples=path_comp_num_samples_scattering, # Removed as it caused TypeError
                         synthetic_array=path_comp_synthetic_array_call,
                         seed=path_comp_seed)
        print("Path computation finished.")
        # Extract data from Paths object and save as NumPy arrays
        paths_data_to_save = {}
        if hasattr(paths, 'targets') and paths.targets is not None:
            paths_data_to_save['targets'] = np.array(paths.targets) # [batch_size, num_rx, num_tx]
        if hasattr(paths, 'vertices') and paths.vertices is not None:
            paths_data_to_save['vertices'] = np.array(paths.vertices) # [batch_size, num_rx, num_tx, max_depth+1, num_paths, 3]
        if hasattr(paths, 'objects') and paths.objects is not None:
            paths_data_to_save['objects'] = np.array(paths.objects) # [batch_size, num_rx, num_tx, max_depth, num_paths]
        if hasattr(paths, 'types') and paths.types is not None:
            paths_data_to_save['types'] = np.array(paths.types) # [batch_size, num_rx, num_tx, max_depth, num_paths]
        # Add other relevant attributes if needed, converting to NumPy
        # For example, if paths.alphas and paths.taus (raw path gains/delays before CIR) are available and useful:
        if hasattr(paths, 'alphas') and paths.alphas is not None: # Complex gains of each path
             paths_data_to_save['path_alphas'] = np.array(paths.alphas)
        if hasattr(paths, 'taus') and paths.taus is not None: # Delays of each path
             paths_data_to_save['path_taus'] = np.array(paths.taus)

        with open(paths_output_file, "wb") as f_paths:
            pickle.dump(paths_data_to_save, f_paths)
        print(f"Extracted propagation paths data saved to {paths_output_file}")

        print("Computing channel impulse responses (CIRs)...")
        a, tau = paths.cir(normalize_delays=True, out_type="numpy") # Corrected: Use paths.cir()
        print("CIR computation finished.")
        
        cir_data = {"gains": a, "delays": tau} # a and tau are already numpy arrays
        with open(cir_output_file, "wb") as f_cir:
            pickle.dump(cir_data, f_cir)
        print(f"Channel impulse responses saved to {cir_output_file}")

        with open(simulation_run_summary_file, "w") as f:
            f.write("Sionna RT Full Simulation Run Summary:\n")
            f.write("-------------------------------------\n")
            f.write(f"Scene file: {xml_file_path}\n")
            f.write(f"Carrier Frequency: {scene.frequency.numpy()/1e9} GHz\n")
            f.write(f"Number of Transmitters: {len(scene.transmitters)}\n")
            f.write(f"Number of Receivers: {len(scene.receivers)}\n")
            f.write(f"PathSolver call parameters: max_depth={path_comp_max_depth}, num_samples_scattering={path_comp_num_samples_scattering.numpy()}, LoS={path_comp_los}, SpecularReflection={path_comp_specular_reflection}, DiffuseReflection={path_comp_diffuse_reflection}, Refraction={path_comp_refraction}, Diffraction={path_comp_diffraction}, SyntheticArrayCall={path_comp_synthetic_array_call}, Seed={path_comp_seed}\n")
            f.write(f"  (Raw paths data in {paths_output_file})\n")
            if paths is not None and hasattr(paths, 'targets') and paths.targets is not None:
                try:
                    # Attempt to describe the content based on expected dimensions
                    if len(paths.targets.shape) >= 3:
                        num_tx_rx_pairs = paths.targets.shape[0] * paths.targets.shape[1] * paths.targets.shape[2]
                        f.write(f"    Paths object contains {num_tx_rx_pairs} potential TX-RX path sets (batch_size * num_rx * num_tx).\n")
                        f.write(f"    Full paths.targets shape: {paths.targets.shape} (batch_size, num_rx, num_tx, num_paths_per_rx_tx_pair)\n")
                    else:
                        f.write(f"    Paths.targets shape: {paths.targets.shape} (unexpected shape, expected at least 3 dimensions for batch_size, num_rx, num_tx)\n")
                except IndexError:
                    f.write(f"    Paths.targets shape: {paths.targets.shape} (Error accessing specific dimensions, shape is unexpected)\n")
            elif paths is not None:
                 f.write(f"    Paths object generated, but paths.targets attribute is missing or None. Check {paths_output_file}.\n")
            else:
                f.write("    No paths object generated or paths object is empty.\n")
            f.write(f"CIRs computed: Yes (gains and delays in {cir_output_file})\n")
            
            effects_list = []
            if path_comp_los: effects_list.append("LoS")
            if path_comp_specular_reflection: effects_list.append("Specular Reflection")
            if path_comp_diffuse_reflection: effects_list.append("Diffuse Reflection (Scattering)")
            if path_comp_refraction: effects_list.append("Refraction/Transmission")
            if path_comp_diffraction: effects_list.append("Diffraction")
            effects_list.append("Absorption (via material conductivity)")
            f.write(f"\nEffects included in simulation attempt: {', '.join(effects_list)}.\n")
        print(f"Simulation run summary saved to {simulation_run_summary_file}")

    except Exception as e:
        error_message = f"Error during simulation run: {e}"
        print(error_message)
        import traceback
        detailed_error = traceback.format_exc()
        print(detailed_error)
        with open(simulation_run_summary_file, "w") as f:
            f.write(error_message + "\n" + detailed_error + "\n")
        return None, None, simulation_run_summary_file # Return summary file even on error

    print("Full simulation script finished successfully.")
    return paths_output_file, cir_output_file, simulation_run_summary_file

if __name__ == '__main__':
    results = run_full_simulation_pathsolver_call()
    if results and results[0] is not None:
        print(f"Simulation completed. Result files: Paths: {results[0]}, CIR: {results[1]}, Summary: {results[2]}")
    elif results and results[2] is not None:
        print(f"Simulation failed but a summary was generated: {results[2]}")
    else:
        print("Simulation failed to complete and no summary was generated.")


