# Development Diary

## May 23, 2026
* **Objective:** Initial project scoping and literature review.
* **Actions Taken:** * Reviewed the provided introductory GIS materials, focusing specifically on spatial data types.
  * Consolidated notes on how vector data represents coordinates via points, lines, and polygons, which is foundational for the enclosing polygon task.
  * Began cross-referencing the Google Earth Engine tutorials with the Google Maps JavaScript API documentation to clarify the target deployment environment.

## May 28, 2026
* **Objective:** Algorithmic research for the enclosing polygon.
* **Actions Taken:** * Defined the core computer science problem: generating a Convex Hull from a set of 2D coordinate points.
  * Evaluated standard algorithms for this task, comparing the time complexities of the Graham Scan, Jarvis March, and the Monotone Chain algorithms.
  * Selected the Monotone Chain approach due to its efficiency and straightforward sorting requirements.
  * Drafted pseudo-code to map the algorithm's logic to latitude and longitude coordinate pairs.

## June 3, 2026
* **Objective:** Environment and infrastructure setup.
* **Actions Taken:** * Configured the local Linux development environment for web testing.
  * Set up version control and initialized the project repository on GitHub.
  * Navigated the Google Cloud Console to understand project creation, API Manager configurations, and credential generation for the Maps JavaScript API.

## June 7, 2026
* **Objective:** Establish baseline environments and prototype core logic.
* **Actions Taken:** * Implemented the Convex Hull algorithm (Monotone Chain) in Python as a standalone script to verify the mathematical modeling of the spatial footprint.
  * Evaluated Google Maps JS API for frontend rendering, but encountered hard billing constraints for API activation.
  * Pivoted to an open-source architecture; successfully created an `index.html` prototype using Leaflet.js and OpenStreetMap tiles to render the base geographical environment locally.
* **Next Actions:** Translate the Python hull logic into JavaScript and dynamically render the outbreak polygon as a vector overlay on the Leaflet map.