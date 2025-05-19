# 5G Localization Challenge – TEKNOFEST 2025

This project was developed for the 5G Localization Challenge organized by Turkcell and TEKNOFEST. The goal was to build a model that can accurately predict the position of mobile devices based on 5G signal measurements.

## Project Objective

Our aim was to estimate the location of a mobile device using path loss values derived from signals emitted by 5G base stations. To achieve this, we implemented two main approaches:

### 1. Physical Modeling (Ray Tracing)

- We utilized Sionna RT to simulate 5G signal propagation in a 3D environment that includes buildings, terrain, and vegetation.
- Electromagnetic interactions such as reflection, diffraction, scattering, and refraction were modeled to generate a realistic radio channel model.
- Transmitter and Receiver objects were added to the scene at appropriate positions.
- From the simulation, we extracted channel impulse responses (CIR) and propagation paths.

File: src/ray_tracing/sionna_ray_tracing_simulation.py

### 2. Path Loss Estimation Model

- Using the Trimesh library, we calculated the physical distances that signals traverse through different environments like buildings and vegetation.
- The path loss model optimizes the following parameters:
  - L_b: Path loss coefficient for buildings
  - L_v: Path loss coefficient for vegetation
  - n: Path-loss exponent
  - C0: Calibration offset
- These parameters were optimized using Huber loss to minimize the difference between measured and predicted path loss values.

File: src/path_loss_modeling/path_loss_model.py

## Technologies Used

- Sionna RT – Ray-tracing-based 3D channel modeling
- Scikit-learn, SciPy – For optimization and model calibration
- Trimesh – For handling and analyzing 3D mesh objects
- Pandas, NumPy, Matplotlib – For data processing and visualization
- PyProj – For geospatial coordinate transformation

## Team Members

- Arda Karakaş
- Alperen Aydın
- Seyit Kaan Güneş
- Baran Akyıldız
- Ahmet Tunahan Yalçın

## Data Privacy Notice

All datasets provided during the competition — including measurement data, base station coordinates, and 3D mesh files — are considered confidential and have not been included in this repository.

## Notes

The repository is structured for clarity, separating the ray tracing simulation from the path loss modeling logic. Each component is documented and designed to be reusable in similar wireless localization projects.
