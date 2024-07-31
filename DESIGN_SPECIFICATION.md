# Design Specification

## Project Overview

This is a Python codebase which models the effect of solar irradiance on a polytunnel. It includes a ray tracing program which simulates sunlight hitting the polytunnel and investigates the irradiance pattern on the ground surface within the tunnel. The project will help in understanding how sunlight is distributed inside a polytunnel, which is essential for optimizing plant growth conditions.

## Repository Structure

```bash
#!/bin/bash
polytunnel-irradiance-model/
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

### 2. sun.py

Calculates sun's altitude and azimuth from specified location.

### 3. ray_tracing.py

Calculates irradiance from trace rays directed from the sun.

### 4. visualisation.py

Constructs grid view of irradiance returned by ray tracing program.

### 5. main.py

Runs complete irradiance model along with visualisations.

## Authors

Taylor Pomfret
