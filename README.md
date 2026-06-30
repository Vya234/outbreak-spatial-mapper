# Spatial Analytics Engine for Disease Outbreaks

## Project Overview
This repository contains a GIS-based web application designed to model and analyze disease outbreaks. It dynamically generates the minimum bounding area (spatial footprint) of an infection cluster using geographic coordinate points and provides real-time geospatial metrics.

## Features & Architecture
* **Algorithmic Engine**: Implements the Monotone Chain convex hull algorithm natively in JavaScript to compute the smallest enclosing polygon around simulated case coordinates.
* **Web-GIS Interface**: An interactive frontend utilizing Leaflet.js and OpenStreetMap tiles for local rendering, complete with drag-and-drop map pins and dynamic threat-level visualization.
* **Geospatial Analytics**: Integrates Turf.js to calculate and display the exact geographic area (km²) of the infection zone in real-time.
* **Data Portability**: Features a CSV export module allowing researchers to seamlessly download active coordinate clusters for secondary data analysis in Python or R.
* **State Management**: Utilizes HTML5 `localStorage` to ensure mapped coordinate arrays persist seamlessly across browser sessions.