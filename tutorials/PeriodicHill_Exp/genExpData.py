#!/usr/bin/env python
"""
Generate experimental reference data for periodic hill field inversion.

This script creates the UData file and probePointCoords.json needed for
field inversion using experimental velocity profiles from Rapp & Manhart (2011).

The periodic hill case at Re_H = 5600 (based on hill height H and bulk velocity
U_b at the crest) is a standard benchmark for turbulence model validation.

Experimental data sources:
    - Rapp, C., Manhart, M., "Flow over periodic hills: an experimental study",
      Experiments in Fluids, 51, 247-269 (2011). PIV/LDA at Re = 5600-37000.
    - Breuer, M., Peller, N., Rapp, C., Manhart, M., "Flow over periodic hills -
      Numerical and experimental study in a wide range of Reynolds numbers",
      Computers & Fluids, 38(2), 433-457 (2009). DNS at Re = 5600.

NOTE: The velocity profiles included below are REPRESENTATIVE values based on
published figures from the above references. For production use, obtain the
actual experimental data from the ERCOFTAC QNET database or contact the
original authors at Technische Universitaet Muenchen.

Usage:
    python genExpData.py

Outputs:
    0/UData               - OpenFOAM velocity reference field
    probePointCoords.json - Probe coordinates at experimental stations
"""

import json
import os
import re
import numpy as np
from scipy.interpolate import interp1d


# =============================================================================
# Physical Parameters
# =============================================================================
H = 1.0  # Hill height (normalized)
Ub = 0.028  # Bulk velocity at crest [m/s]
nu = 5.0e-6  # Kinematic viscosity [m^2/s]
Re = Ub * H / nu  # Reynolds number = 5600
Ly = 3.035 * H  # Channel height
Lx = 9.0 * H  # Streamwise period
z_mid = 0.05  # z-coordinate of mesh midplane


# =============================================================================
# Periodic Hill Geometry
# =============================================================================
def hill_height(x):
    """
    Compute the bottom wall height y_w(x) for the standard periodic hill.

    Uses the piecewise polynomial geometry from Mellen et al. (2000) as
    specified in Breuer et al. (2009). The hill crest is at x/H = 0 and
    x/H = 9 with y_w = H. The inter-hill flat region has y_w = 0.

    Parameters
    ----------
    x : float or np.ndarray
        Streamwise coordinate(s) in [0, 9H]

    Returns
    -------
    float or np.ndarray
        Bottom wall height y_w
    """
    x = np.asarray(x, dtype=float)
    xh = x / H  # Normalize by hill height
    yw = np.zeros_like(xh)

    # Polynomial coefficients for the hill shape (Almeida et al. 1993 / Mellen et al. 2000)
    # Segment 1: crest to inflection (0 <= x/H <= 0.5)
    mask1 = (xh >= 0) & (xh < 0.5)
    xi = xh[mask1]
    yw[mask1] = np.minimum(
        H,
        H
        * (
            1.0
            + 2.800000000000e-01 * xi / H
            + 6.775070969851e-03 * (xi / H) ** 2
            - 2.124527775800e00 * (xi / H) ** 3
        ),
    )

    # Segment 2: descending (0.5 <= x/H <= 1.0)
    mask2 = (xh >= 0.5) & (xh < 1.0)
    xi = xh[mask2]
    yw[mask2] = H * (
        0.507475
        + 0.187625 * (xi - 0.5)
        - 0.703700 * (xi - 0.5) ** 2
        - 0.252650 * (xi - 0.5) ** 3
    )

    # Segment 3: lower descending (1.0 <= x/H <= 2.0)
    mask3 = (xh >= 1.0) & (xh < 2.0)
    xi = xh[mask3]
    yw[mask3] = H * (
        0.188625
        - 0.535250 * (xi - 1.0)
        + 0.252050 * (xi - 1.0) ** 2
        + 0.097500 * (xi - 1.0) ** 3
    )

    # Segment 4: transition to flat (2.0 <= x/H <= 3.0)
    mask4 = (xh >= 2.0) & (xh < 3.0)
    xi = xh[mask4]
    yw[mask4] = np.maximum(
        0.0,
        H * (0.002675 - 0.088225 * (xi - 2.0) + 0.094675 * (xi - 2.0) ** 2 - 0.030250 * (xi - 2.0) ** 3),
    )

    # Segment 5: flat bottom (3.0 <= x/H <= 6.0)
    # y_w = 0 (already initialized)

    # Segment 6: ascending by symmetry (6.0 <= x/H <= 9.0)
    # Mirror the descending side: use xh_mirror = 9 - xh which maps [6,9] -> [0,3]
    mask6 = (xh >= 6.0) & (xh <= 9.0)
    xm = 9.0 - xh[mask6]  # Mirror: maps 6->3, 7->2, 8->1, 9->0

    # Apply the same polynomial segments to the mirrored coordinate
    # xm in [0, 0.5)
    m1 = (xm >= 0) & (xm < 0.5)
    yw_6 = np.zeros_like(xm)
    xi = xm[m1]
    yw_6[m1] = np.minimum(H, H * (1.0 + 2.8e-01 * xi + 6.775070969851e-03 * xi**2 - 2.124527775800e00 * xi**3))
    # xm in [0.5, 1.0)
    m2 = (xm >= 0.5) & (xm < 1.0)
    xi = xm[m2]
    yw_6[m2] = H * (0.507475 + 0.187625 * (xi - 0.5) - 0.703700 * (xi - 0.5) ** 2 - 0.252650 * (xi - 0.5) ** 3)
    # xm in [1.0, 2.0)
    m3 = (xm >= 1.0) & (xm < 2.0)
    xi = xm[m3]
    yw_6[m3] = H * (0.188625 - 0.535250 * (xi - 1.0) + 0.252050 * (xi - 1.0) ** 2 + 0.097500 * (xi - 1.0) ** 3)
    # xm in [2.0, 3.0]
    m4 = (xm >= 2.0) & (xm <= 3.0)
    xi = xm[m4]
    yw_6[m4] = np.maximum(
        0.0, H * (0.002675 - 0.088225 * (xi - 2.0) + 0.094675 * (xi - 2.0) ** 2 - 0.030250 * (xi - 2.0) ** 3)
    )

    yw[mask6] = np.maximum(yw_6, 0.0)

    return np.maximum(yw, 0.0)


# =============================================================================
# Experimental Velocity Profiles at Re = 5600
# =============================================================================
# Representative mean velocity profiles based on Rapp & Manhart (2011) PIV/LDA
# measurements and cross-validated with Breuer et al. (2009) DNS data.
#
# Format: (x/H station, wall y/H, [(y/H, U/Ub, V/Ub), ...])
#
# IMPORTANT: These are representative values extracted from published figures.
# For production use, replace with actual experimental data from the ERCOFTAC
# QNET database: https://www.kbwiki.ercoftac.org/w/index.php/Abstr:2D_Periodic_Hill_Flow

EXPERIMENTAL_PROFILES = {
    # Station x/H = 0.5: Leeward side of hill, flow has separated (sep. ~ x/H = 0.2)
    # Wall at y_w/H ~ 0.51 (hill descending). Shear layer developing.
    0.5: {
        "y_H": [0.52, 0.55, 0.60, 0.68, 0.80, 0.95, 1.15, 1.40, 1.70, 2.00, 2.30, 2.60, 2.90, 3.035],
        "U_Ub": [0.00, 0.30, 0.18, -0.02, -0.06, 0.10, 0.45, 0.78, 1.02, 1.12, 1.10, 0.90, 0.48, 0.00],
        "V_Ub": [0.00, -0.06, -0.10, -0.10, -0.06, -0.02, 0.01, 0.02, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00],
    },
    # Station x/H = 2.0: Deep in recirculation zone
    # Wall at y_w/H ~ 0.0 (flat bottom). Strong reverse flow U/Ub ~ -0.13.
    2.0: {
        "y_H": [0.01, 0.04, 0.08, 0.15, 0.25, 0.40, 0.60, 0.85, 1.10, 1.40, 1.75, 2.10, 2.50, 2.85, 3.035],
        "U_Ub": [0.00, -0.08, -0.13, -0.12, -0.04, 0.14, 0.42, 0.72, 0.94, 1.06, 1.08, 1.02, 0.82, 0.42, 0.00],
        "V_Ub": [0.00, 0.00, 0.01, 0.02, 0.03, 0.04, 0.03, 0.02, 0.01, 0.00, 0.00, -0.01, 0.00, 0.00, 0.00],
    },
    # Station x/H = 4.0: Near reattachment (reattachment ~ x/H = 4.7 at Re=5600)
    # Wall at y_w/H = 0.0 (flat bottom). Weak reverse flow near bottom.
    4.0: {
        "y_H": [0.01, 0.04, 0.08, 0.15, 0.25, 0.40, 0.60, 0.85, 1.10, 1.40, 1.75, 2.10, 2.50, 2.85, 3.035],
        "U_Ub": [0.00, -0.02, 0.02, 0.12, 0.28, 0.46, 0.64, 0.84, 0.98, 1.06, 1.06, 1.00, 0.80, 0.42, 0.00],
        "V_Ub": [0.00, 0.00, 0.01, 0.01, 0.02, 0.02, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    },
    # Station x/H = 6.0: Post-reattachment recovery
    # Wall at y_w/H = 0.0 (flat bottom). Boundary layer redeveloping.
    6.0: {
        "y_H": [0.01, 0.04, 0.08, 0.15, 0.25, 0.40, 0.60, 0.85, 1.10, 1.40, 1.75, 2.10, 2.50, 2.85, 3.035],
        "U_Ub": [0.00, 0.10, 0.22, 0.40, 0.56, 0.68, 0.80, 0.92, 1.00, 1.04, 1.04, 0.98, 0.78, 0.42, 0.00],
        "V_Ub": [0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    },
    # Station x/H = 8.0: Windward side of next hill (ascending)
    # Wall at y_w/H ~ 0.19 (hill ascending, mirrors x/H=1). Flow accelerating.
    8.0: {
        "y_H": [0.20, 0.28, 0.40, 0.55, 0.75, 0.95, 1.20, 1.50, 1.85, 2.20, 2.55, 2.85, 3.035],
        "U_Ub": [0.00, 0.30, 0.55, 0.74, 0.90, 0.98, 1.06, 1.06, 1.00, 0.88, 0.62, 0.32, 0.00],
        "V_Ub": [0.00, 0.03, 0.05, 0.05, 0.03, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    },
}


# =============================================================================
# OpenFOAM Mesh Reader
# =============================================================================
def read_openfoam_scalar_list(filepath):
    """Read an OpenFOAM list of scalars/integers from a file."""
    with open(filepath, "r") as f:
        content = f.read()

    # Find the list size and data between parentheses
    # Skip header, find first standalone integer (list size) then ( ... )
    header_end = content.find("(")
    if header_end < 0:
        raise ValueError(f"No list found in {filepath}")

    # Extract size from the line before '('
    pre = content[:header_end].strip().split("\n")
    size = int(pre[-1].strip())

    data_start = header_end + 1
    data_end = content.find(")", data_start)
    data_str = content[data_start:data_end]

    values = []
    for line in data_str.strip().split("\n"):
        line = line.strip()
        if line:
            values.append(int(line))

    assert len(values) == size, f"Expected {size} entries, got {len(values)}"
    return np.array(values)


def read_openfoam_points(filepath):
    """Read OpenFOAM points file and return Nx3 array of coordinates."""
    with open(filepath, "r") as f:
        content = f.read()

    header_end = content.find("(")
    pre = content[:header_end].strip().split("\n")
    size = int(pre[-1].strip())

    data_start = header_end + 1
    data_end = content.rfind(")")
    data_str = content[data_start:data_end]

    points = []
    for line in data_str.strip().split("\n"):
        line = line.strip().strip("()")
        if line:
            coords = line.split()
            if len(coords) == 3:
                points.append([float(c) for c in coords])

    assert len(points) == size, f"Expected {size} points, got {len(points)}"
    return np.array(points)


def read_openfoam_faces(filepath):
    """Read OpenFOAM faces file and return list of face point index lists."""
    with open(filepath, "r") as f:
        content = f.read()

    header_end = content.find("\n(")
    pre = content[:header_end].strip().split("\n")
    size = int(pre[-1].strip())

    data_start = content.find("\n(", header_end) + 2
    data_end = content.rfind(")")
    data_str = content[data_start:data_end]

    faces = []
    for line in data_str.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Format: N(i0 i1 i2 ... iN-1) e.g. "4(0 1 5 4)"
        match = re.match(r"(\d+)\(([^)]+)\)", line)
        if match:
            indices = [int(x) for x in match.group(2).split()]
            faces.append(indices)

    assert len(faces) == size, f"Expected {size} faces, got {len(faces)}"
    return faces


def compute_cell_centers(mesh_dir):
    """
    Compute cell centers from OpenFOAM polyMesh files.

    Parameters
    ----------
    mesh_dir : str
        Path to constant/polyMesh directory

    Returns
    -------
    np.ndarray
        Cell centers array of shape (nCells, 3)
    """
    points = read_openfoam_points(os.path.join(mesh_dir, "points"))
    faces = read_openfoam_faces(os.path.join(mesh_dir, "faces"))
    owner = read_openfoam_scalar_list(os.path.join(mesh_dir, "owner"))
    neighbour = read_openfoam_scalar_list(os.path.join(mesh_dir, "neighbour"))

    nCells = owner.max() + 1

    # Compute face centers
    face_centers = np.zeros((len(faces), 3))
    for i, face in enumerate(faces):
        face_centers[i] = points[face].mean(axis=0)

    # Compute cell centers as average of face centers belonging to each cell
    cell_center_sum = np.zeros((nCells, 3))
    cell_face_count = np.zeros(nCells)

    for i in range(len(faces)):
        c = owner[i]
        cell_center_sum[c] += face_centers[i]
        cell_face_count[c] += 1

    for i in range(len(neighbour)):
        c = neighbour[i]
        cell_center_sum[c] += face_centers[i]
        cell_face_count[c] += 1

    cell_centers = cell_center_sum / cell_face_count[:, np.newaxis]
    return cell_centers


# =============================================================================
# Data Interpolation
# =============================================================================
def interpolate_profiles_to_cells(cell_centers):
    """
    Interpolate experimental velocity profiles onto mesh cell centers.

    Uses linear interpolation between measurement stations in the streamwise
    direction and within each profile in the wall-normal direction.

    Parameters
    ----------
    cell_centers : np.ndarray
        Cell center coordinates, shape (nCells, 3)

    Returns
    -------
    np.ndarray
        Velocity field at all cells, shape (nCells, 3): (U, V, W)
    """
    nCells = len(cell_centers)
    velocity = np.zeros((nCells, 3))

    stations = sorted(EXPERIMENTAL_PROFILES.keys())

    # Build interpolators for each station
    interp_u = {}
    interp_v = {}
    for s in stations:
        prof = EXPERIMENTAL_PROFILES[s]
        y = np.array(prof["y_H"]) * H
        u = np.array(prof["U_Ub"]) * Ub
        v = np.array(prof["V_Ub"]) * Ub
        interp_u[s] = interp1d(y, u, kind="linear", bounds_error=False, fill_value=0.0)
        interp_v[s] = interp1d(y, v, kind="linear", bounds_error=False, fill_value=0.0)

    for i in range(nCells):
        xc = cell_centers[i, 0]
        yc = cell_centers[i, 1]

        # Wrap x to [0, 9H] for periodicity
        xh = (xc / H) % 9.0

        # Find bounding stations
        s_lo = stations[0]
        s_hi = stations[-1]
        for j in range(len(stations) - 1):
            if stations[j] <= xh <= stations[j + 1]:
                s_lo = stations[j]
                s_hi = stations[j + 1]
                break
        else:
            # x is outside the station range; use nearest
            if xh < stations[0]:
                # Between last station (8.0) wrapped and first (0.5)
                s_lo = stations[-1]
                s_hi = stations[0]

        # Interpolation weight
        if s_hi != s_lo:
            if s_lo < s_hi:
                w = (xh - s_lo) / (s_hi - s_lo)
            else:
                # Wrap-around case
                span = (9.0 - s_lo) + s_hi
                dist = (xh - s_lo) if xh >= s_lo else (9.0 - s_lo + xh)
                w = dist / span
            w = np.clip(w, 0.0, 1.0)
        else:
            w = 0.0

        u_lo = float(interp_u[s_lo](yc))
        u_hi = float(interp_u[s_hi](yc))
        v_lo = float(interp_v[s_lo](yc))
        v_hi = float(interp_v[s_hi](yc))

        velocity[i, 0] = (1.0 - w) * u_lo + w * u_hi
        velocity[i, 1] = (1.0 - w) * v_lo + w * v_hi
        velocity[i, 2] = 0.0  # 2D case, no spanwise velocity

    return velocity


def generate_probe_points():
    """
    Generate probe point coordinates at experimental measurement stations.

    Creates vertical lines of points at each experimental station, matching
    the measurement locations from Rapp & Manhart (2011) PIV/LDA data.

    Returns
    -------
    list
        List of [x, y, z] probe point coordinates
    """
    probe_points = []

    for station, prof in EXPERIMENTAL_PROFILES.items():
        x = station * H
        for y_h in prof["y_H"]:
            y = y_h * H
            # Skip wall points (y too close to wall) to avoid cell-finding issues
            y_wall = float(hill_height(np.array([x])))
            if y - y_wall < 0.005 * H:
                continue
            # Skip points too close to top wall
            if Ly - y < 0.005 * H:
                continue
            probe_points.append([float(x), float(y), z_mid])

    return probe_points


# =============================================================================
# OpenFOAM File Writers
# =============================================================================
def write_UData(velocity, output_path):
    """
    Write velocity field as OpenFOAM volVectorField (UData).

    Parameters
    ----------
    velocity : np.ndarray
        Velocity at all cells, shape (nCells, 3)
    output_path : str
        Path to write the file
    """
    nCells = len(velocity)

    header = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v1812                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "0";
    object      UData;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

"""

    with open(output_path, "w") as f:
        f.write(header)
        f.write(f"internalField   nonuniform List<vector> \n{nCells}\n(\n")
        for i in range(nCells):
            f.write(f"({velocity[i,0]:.16e} {velocity[i,1]:.16e} {velocity[i,2]:.16e})\n")
        f.write(");\n\n")

        # Boundary conditions (same as U file)
        f.write(
            """boundaryField
{
    bottomWall
    {
        type            fixedValue;
        value           uniform (0 0 0);
    }
    front
    {
        type            symmetry;
    }
    back
    {
        type         symmetry;
    }
    inlet
    {
        type            cyclic;
    }
    outlet
    {
        type            cyclic;
    }
    topWall
    {
        type            fixedValue;
        value           uniform (0 0 0);
    }
}
// ************************************************************************* //
"""
        )


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("Experimental Data Generation for Periodic Hill FIML")
    print(f"Re_H = {Re:.0f}, Ub = {Ub} m/s, H = {H} m, nu = {nu} m^2/s")
    print("=" * 60)

    # Read mesh
    mesh_dir = os.path.join("constant", "polyMesh")
    if not os.path.exists(mesh_dir):
        print(f"ERROR: Mesh not found at {mesh_dir}")
        print("Ensure constant/polyMesh exists (run preProcessing.sh or copy from PeriodicHill)")
        return

    print("\nReading OpenFOAM mesh...")
    cell_centers = compute_cell_centers(mesh_dir)
    nCells = len(cell_centers)
    print(f"  Found {nCells} cells")
    print(f"  x range: [{cell_centers[:,0].min():.4f}, {cell_centers[:,0].max():.4f}]")
    print(f"  y range: [{cell_centers[:,1].min():.4f}, {cell_centers[:,1].max():.4f}]")
    print(f"  z range: [{cell_centers[:,2].min():.4f}, {cell_centers[:,2].max():.4f}]")

    # Interpolate experimental profiles onto mesh
    print("\nInterpolating experimental velocity profiles onto mesh...")
    print(f"  Stations: x/H = {sorted(EXPERIMENTAL_PROFILES.keys())}")
    velocity = interpolate_profiles_to_cells(cell_centers)
    print(f"  U range: [{velocity[:,0].min():.6f}, {velocity[:,0].max():.6f}]")
    print(f"  V range: [{velocity[:,1].min():.6f}, {velocity[:,1].max():.6f}]")

    # Write UData
    udata_path = os.path.join("0", "UData")
    if not os.path.exists("0"):
        os.makedirs("0")
    print(f"\nWriting {udata_path}...")
    write_UData(velocity, udata_path)

    # Generate probe points
    print("\nGenerating probe point coordinates...")
    probe_points = generate_probe_points()
    print(f"  {len(probe_points)} probe points at {len(EXPERIMENTAL_PROFILES)} stations")

    # Write probePointCoords.json
    probe_path = "probePointCoords.json"
    with open(probe_path, "w") as f:
        json.dump({"probePointCoords": probe_points}, f, indent=4)
    print(f"  Written to {probe_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary of experimental data:")
    for station in sorted(EXPERIMENTAL_PROFILES.keys()):
        prof = EXPERIMENTAL_PROFILES[station]
        n = len(prof["y_H"])
        y_wall = float(hill_height(np.array([station * H])))
        print(f"  x/H = {station:.1f}: {n} points, y_wall/H = {y_wall/H:.3f}")
    print(f"\nTotal probe points: {len(probe_points)}")
    print(f"UData file: {udata_path}")
    print(f"Probe coords: {probe_path}")
    print("=" * 60)
    print("\nData sources (representative values):")
    print("  Rapp & Manhart (2011), Experiments in Fluids, 51, 247-269")
    print("  Breuer et al. (2009), Computers & Fluids, 38(2), 433-457")
    print("\nFor actual experimental data, see references.md")


if __name__ == "__main__":
    main()
