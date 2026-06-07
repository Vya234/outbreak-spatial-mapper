def compute_hull(points):
    """Computes the convex hull of a set of 2D points."""
    # Sort points lexicographically
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    # Cross product to determine turn direction
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]

# Test with some dummy lat/lng coordinates near campus
sample_points = [
    (22.3149, 87.3105), 
    (22.3155, 87.3151), 
    (22.3121, 87.3120), 
    (22.3160, 87.3110),
    (22.3135, 87.3135)
]

hull = compute_hull(sample_points)
print("Input Points:", sample_points)
print("Convex Hull Points:", hull)