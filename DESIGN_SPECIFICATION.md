# Design Specification

## Project Overview

This is a Python codebase which models the effect of solar irradiance on a polytunnel. It includes a ray tracing program which simulates direct sunlight hitting the poly tunnel and investigates the irradiance pattern on the ground surface within the tunnel. The project will help in understanding how sunlight distributes inside a poly tunnel, which is essential for optimizing plant growth conditions.

## Repository Structure

```
poly-tunnel-irradiance/
├── README.md
├── DESIGN_SPECIFICATION.md
├── FUNCTIONAL_SPECIFICATION.md
├── src/
├── tests/
├── data/
├── docs/
```

## Modules 

### 1. geometry.py

Defines geometry of polytunnel and ground surface. 

### 2. ray_tracing.py

Calculates irradiance from trace rays directed from the sun. 

### 3. visualisation.py

Constructs grid view of irradiance returned by ray tracing program.

### 4. main.py 

Runs complete irradiance model.

## Authors

Taylor Pomfret
