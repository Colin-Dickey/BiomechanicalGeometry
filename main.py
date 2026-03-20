import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os

FILEPATHS = [
    "femurFolder/m2LeftFemur.obj",
    "femurFolder/m2RightFemur.obj",
]

N                   = 1600  # neighbour nodes for sphere fit
N_CYL               = 800   # neighbour nodes for cylinder fit
N_NECK              = 300   # neighbour nodes for neck ring
NECK_RING_OFFSET_MM = 1     # extra mm added to distance from femoral head centre
CYL_CUTOFF_MM       = 5.0   # mm to ignore from bottom for cylinder seed search

# ---------------------------------------------------------------
# Functions
# ---------------------------------------------------------------

def readObjVertices(filepath):
    vertices = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
    return np.array(vertices)

def sphere_residuals(params, points):
    cx, cy, cz, r = params
    distances = np.sqrt((points[:, 0] - cx)**2 +
                        (points[:, 1] - cy)**2 +
                        (points[:, 2] - cz)**2)
    return distances - r

def cylinder_residuals_3d(params, points):
    cx, cy, cz, ax, ay, az, r = params
    axis = np.array([ax, ay, az])
    axis = axis / np.linalg.norm(axis)
    p = points - np.array([cx, cy, cz])
    dot = p @ axis
    projection = np.outer(dot, axis)
    perp = p - projection
    distances = np.linalg.norm(perp, axis=1)
    return distances - r

def get_n_neighbours(coords, seed_index, n):
    collected = [seed_index]
    collected_set = set(collected)
    while len(collected) < n:
        dists = np.linalg.norm(coords - coords[seed_index], axis=1)
        dists[list(collected_set)] = np.inf
        next_node = int(np.argmin(dists))
        collected.append(next_node)
        collected_set.add(next_node)
    return np.array(collected)

def fit_sphere(targets):
    centroid  = targets.mean(axis=0)
    initial_r = np.mean(np.linalg.norm(targets - centroid, axis=1))
    x0        = [centroid[0], centroid[1], centroid[2], initial_r]
    result    = least_squares(sphere_residuals, x0, args=(targets,))
    cx, cy, cz, r = result.x
    print(f"  Femoral head centre: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
    print(f"  Femoral head radius: {r:.3f}")
    print(f"  Sphere residual RMS: {np.sqrt(np.mean(result.fun**2)):.4f}")
    return cx, cy, cz, r

def fit_cylinder_3d(targets):
    cx0 = targets[:, 0].mean()
    cy0 = targets[:, 1].mean()
    cz0 = targets[:, 2].mean()
    r0  = np.mean(np.sqrt((targets[:, 0] - cx0)**2 +
                           (targets[:, 1] - cy0)**2))
    x0     = [cx0, cy0, cz0, 0, 0, 1, r0]
    result = least_squares(cylinder_residuals_3d, x0, args=(targets,))
    cx, cy, cz, ax, ay, az, r = result.x
    axis = np.array([ax, ay, az])
    axis = axis / np.linalg.norm(axis)
    print(f"  Cylinder centre: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
    print(f"  Cylinder axis:   ({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f})")
    print(f"  Cylinder radius: {r:.3f}")
    print(f"  Cylinder residual RMS: {np.sqrt(np.mean(result.fun**2)):.4f}")
    return cx, cy, cz, axis, r

def find_sphere_opposite_point(sphere_centre, sphere_radius, seed_point):
    sphere_centre = np.array(sphere_centre)
    seed_point    = np.array(seed_point)
    direction     = sphere_centre - seed_point
    direction     = direction / np.linalg.norm(direction)
    return sphere_centre + sphere_radius * direction

def get_neck_ring(coords, seed_index, sphere_centre, n, tolerance=2.0, offset_mm=0.0):
    sphere_centre  = np.array(sphere_centre)
    dist_to_centre = np.linalg.norm(coords - sphere_centre, axis=1)
    seed_dist      = dist_to_centre[seed_index] + offset_mm
    print(f"  Neck ring target distance from head centre: {seed_dist:.3f}mm (offset: {offset_mm}mm)")
    on_ring_mask  = np.abs(dist_to_centre - seed_dist) < tolerance
    collected     = [seed_index]
    collected_set = set(collected)
    while len(collected) < n:
        dists = np.linalg.norm(coords - coords[seed_index], axis=1)
        dists[list(collected_set)] = np.inf
        dists[~on_ring_mask]       = np.inf
        next_node = int(np.argmin(dists))
        if dists[next_node] == np.inf:
            print(f"  Ran out of ring nodes at {len(collected)}")
            break
        collected.append(next_node)
        collected_set.add(next_node)
    return np.array(collected)

def process_femur(filepath):
    """Run all geometry fitting for a single femur, return results dict."""
    print(f"\n--- Processing: {filepath} ---")
    obj_name = os.path.splitext(os.path.basename(filepath))[0]

    coords   = readObjVertices(filepath)
    minimumZ = min(coords[:, 2])
    coords[:, 2] = coords[:, 2] + minimumZ * -1
    print(f"  Loaded {coords.shape[0]} nodes")

    # Sphere seed direction
    if 'right' in obj_name.lower():
        sphere_direction = np.array([1, -1, 1], dtype=float)
    else:
        sphere_direction = np.array([-1, -1, 1], dtype=float)
    sphere_direction = sphere_direction / np.linalg.norm(sphere_direction)

    projections  = coords @ sphere_direction
    corner_index = int(np.argmax(projections))
    maxz_index   = int(np.argmax(coords[:, 2]))

    # Sphere fit
    targets_indices = get_n_neighbours(coords, corner_index, N)
    targets         = coords[targets_indices]
    cx_s, cy_s, cz_s, r_s = fit_sphere(targets)
    sphere_centre = np.array([cx_s, cy_s, cz_s])

    # Opposite point and neck seed
    opposite_on_sphere = find_sphere_opposite_point(sphere_centre, r_s, coords[corner_index])
    dists_to_opposite  = np.linalg.norm(coords - opposite_on_sphere, axis=1)
    neck_seed_index    = int(np.argmin(dists_to_opposite))

    # Neck ring
    neck_ring_indices = get_neck_ring(
        coords, neck_seed_index, sphere_centre, N_NECK,
        tolerance=2.0, offset_mm=NECK_RING_OFFSET_MM
    )
    neck_ring     = coords[neck_ring_indices]
    neck_centroid = neck_ring.mean(axis=0)
    print(f"  Neck centroid: {neck_centroid}")

    # Cylinder seed and fit
    cutoff_z          = CYL_CUTOFF_MM
    shaft_mask        = coords[:, 2] > cutoff_z
    shaft_indices     = np.where(shaft_mask)[0]
    shaft_only_coords = coords[shaft_mask]
    cyl_seed_local    = int(np.argmin(shaft_only_coords[:, 2]))
    cyl_seed_index    = int(shaft_indices[cyl_seed_local])

    cyl_targets_indices_local = get_n_neighbours(shaft_only_coords, cyl_seed_local, N_CYL)
    cyl_targets               = shaft_only_coords[cyl_targets_indices_local]
    cx_c, cy_c, cz_c, cyl_axis, r_c = fit_cylinder_3d(cyl_targets)

    return {
        'obj_name':          obj_name,
        'coords':            coords,
        'targets':           targets,
        'cyl_targets':       cyl_targets,
        'neck_ring':         neck_ring,
        'neck_centroid':     neck_centroid,
        'corner_index':      corner_index,
        'maxz_index':        maxz_index,
        'cyl_seed_index':    cyl_seed_index,
        'opposite_on_sphere': opposite_on_sphere,
        'neck_seed_coords':  coords[neck_seed_index],
        'sphere_centre':     sphere_centre,
        'r_s':               r_s,
        'cx_s': cx_s, 'cy_s': cy_s, 'cz_s': cz_s,
        'cx_c': cx_c, 'cy_c': cy_c, 'cz_c': cz_c,
        'cyl_axis':          cyl_axis,
        'r_c':               r_c,
        'cutoff_z':          cutoff_z,
    }

def plot_femur(ax, d):
    """Plot all geometry for one femur onto a given axes."""
    coords   = d['coords']
    sampled  = coords[::5]
    cutoff_z = d['cutoff_z']

    ignored_coords = coords[coords[:, 2] <= cutoff_z]

    # Mesh
    ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2],
               c='steelblue', s=1, alpha=0.3, label='Mesh nodes')

    # Ignored
    ax.scatter(ignored_coords[::5, 0], ignored_coords[::5, 1], ignored_coords[::5, 2],
               c='grey', s=1, alpha=0.2, label=f'Ignored (<{cutoff_z:.1f}mm)')

    # Femoral head points
    ax.scatter(d['targets'][:, 0], d['targets'][:, 1], d['targets'][:, 2],
               c='red', s=20, zorder=5, label=f'Head points (n={N})')

    # Cylinder points
    ax.scatter(d['cyl_targets'][:, 0], d['cyl_targets'][:, 1], d['cyl_targets'][:, 2],
               c='orange', s=20, zorder=5, label=f'Cylinder points (n={N_CYL})')

    # Neck ring
    ax.scatter(d['neck_ring'][:, 0], d['neck_ring'][:, 1], d['neck_ring'][:, 2],
               c='purple', s=20, zorder=5, label=f'Neck ring (n={len(d["neck_ring"])})')

    # Neck centroid
    nc = d['neck_centroid']
    ax.scatter(nc[0], nc[1], nc[2],
               c='purple', s=150, zorder=6, edgecolors='black', label='Neck centroid')

    # Neck axis line
    sc       = d['sphere_centre']
    line_dir = nc - sc
    line_dir = line_dir / np.linalg.norm(line_dir)
    p_start  = sc - line_dir * 20
    p_end    = sc + line_dir * 150
    ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], [p_start[2], p_end[2]],
            c='purple', linewidth=2, linestyle='--', label='Neck axis')

    # Seed nodes
    for idx in [d['corner_index'], d['maxz_index'], d['cyl_seed_index']]:
        ax.scatter(coords[idx, 0], coords[idx, 1], coords[idx, 2],
                   c='yellow', s=150, zorder=6, edgecolors='black')
    ax.scatter([], [], [], c='yellow', edgecolors='black', s=150, label='Seed nodes')

    # Opposite sphere point and neck seed
    op = d['opposite_on_sphere']
    ax.scatter(op[0], op[1], op[2], c='cyan', s=150, zorder=6,
               marker='*', label='Opposite sphere point')
    ns = d['neck_seed_coords']
    ax.scatter(ns[0], ns[1], ns[2], c='magenta', s=150, zorder=6,
               edgecolors='black', label='Neck seed')

    # Fitted sphere
    u  = np.linspace(0, 2 * np.pi, 30)
    v  = np.linspace(0, np.pi, 30)
    sx = d['cx_s'] + d['r_s'] * np.outer(np.cos(u), np.sin(v))
    sy = d['cy_s'] + d['r_s'] * np.outer(np.sin(u), np.sin(v))
    sz = d['cz_s'] + d['r_s'] * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(sx, sy, sz, color='green', alpha=0.2, linewidth=0.5,
                      label=f'Sphere R={d["r_s"]:.2f}mm')
    ax.scatter(d['cx_s'], d['cy_s'], d['cz_s'], c='green', s=100,
               zorder=5, label='Head centre')

    # Fitted cylinder
    cyl_centre     = np.array([d['cx_c'], d['cy_c'], d['cz_c']])
    cyl_axis       = d['cyl_axis']
    r_c            = d['r_c']
    cyl_length     = d['cyl_targets'][:, 2].max() - d['cyl_targets'][:, 2].min()
    axis_extension = 150

    perp1 = np.cross(cyl_axis, [0, 0, 1])
    if np.linalg.norm(perp1) < 1e-6:
        perp1 = np.cross(cyl_axis, [0, 1, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(cyl_axis, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)

    theta = np.linspace(0, 2 * np.pi, 30)
    t     = np.linspace(-cyl_length / 2, cyl_length / 2, 30)
    x_cyl = np.zeros((len(t), len(theta)))
    y_cyl = np.zeros((len(t), len(theta)))
    z_cyl = np.zeros((len(t), len(theta)))

    for i, ti in enumerate(t):
        for j, tj in enumerate(theta):
            point = (cyl_centre + ti * cyl_axis +
                     r_c * np.cos(tj) * perp1 +
                     r_c * np.sin(tj) * perp2)
            x_cyl[i, j] = point[0]
            y_cyl[i, j] = point[1]
            z_cyl[i, j] = point[2]

    ax.plot_wireframe(x_cyl, y_cyl, z_cyl, color='orange', alpha=0.2,
                      linewidth=0.5, label=f'Cylinder R={r_c:.2f}mm')

    p1 = cyl_centre - cyl_axis * (cyl_length / 2 + axis_extension)
    p2 = cyl_centre + cyl_axis * (cyl_length / 2 + axis_extension)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            c='orange', linewidth=2, label='Cylinder axis')

    # Normalise axes
    max_range = np.array([
        coords[:, 0].max() - coords[:, 0].min(),
        coords[:, 1].max() - coords[:, 1].min(),
        coords[:, 2].max() - coords[:, 2].min()
    ]).max() / 2

    mid_x = (coords[:, 0].max() + coords[:, 0].min()) / 2
    mid_y = (coords[:, 1].max() + coords[:, 1].min()) / 2
    mid_z = (coords[:, 2].max() + coords[:, 2].min()) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper left', fontsize=6)
    ax.set_title(f'{d["obj_name"]} — Sphere R={d["r_s"]:.2f}mm | Cylinder R={d["r_c"]:.2f}mm', pad=20, y=0.95)

results = [process_femur(fp) for fp in FILEPATHS]

fig = plt.figure(figsize=(18, 9))

for i, d in enumerate(results):
    ax = fig.add_subplot(1, len(results), i + 1, projection='3d')
    plot_femur(ax, d)

plt.tight_layout()

os.makedirs('graphImages', exist_ok=True)
combined_name = '_'.join([r['obj_name'] for r in results])
plt.savefig(f'graphImages/{combined_name}_fit.png', dpi=150, bbox_inches='tight')
plt.show()