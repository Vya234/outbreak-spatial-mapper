# Spatial Footprint Mapper for Disease Outbreaks

## Project Overview
This repository contains the foundational prototyping for a GIS-based web application designed to model disease outbreaks. The core objective is to generate and visualize the minimum bounding area (spatial footprint) of an infection cluster using a set of geographic coordinate points.

## Architecture & Current Prototypes
The development is currently split into two modules:
1. **Algorithmic Engine (`convex_hull.py`)**: A Python implementation of the Monotone Chain algorithm. This serves as the mathematical foundation to compute the smallest enclosing polygon around simulated case coordinates.
2. **Web-GIS Interface (`index.html`)**: A local frontend prototype utilizing Leaflet.js and OpenStreetMap tiles to render the base geographical environment.

## Next Steps
- Translate the Python convex hull logic into JavaScript.
- Dynamically render the calculated outbreak polygon as a vector overlay on the Leaflet map instance.