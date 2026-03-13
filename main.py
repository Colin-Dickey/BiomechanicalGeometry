import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os

filepath = "femurFolder/m1RightFemur.obj"
N             = 1600  # neighbour nodes for sphere fit
N_CYL         = 1600   # neighbour nodes for cylinder fit
CYL_CUTOFF_MM = 5.0   # mm to ignore from bottom for cylinder seed search

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
    """
    Residuals for a cylinder with free axis direction.
    params: [cx, cy, cz, ax, ay, az, r]
    c = point on axis, a = axis direction vector, r = radius
    """
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
    """
    Starting from seed_index, repeatedly find the next nearest unvisited node.
    """
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
    centroid = targets.mean(axis=0)
    initial_r = np.mean(np.linalg.norm(targets - centroid, axis=1))
    x0 = [centroid[0], centroid[1], centroid[2], initial_r]
    result = least_squares(sphere_residuals, x0, args=(targets,))
    cx, cy, cz, r = result.x
    print(f"Femoral head centre: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
    print(f"Femoral head radius: {r:.3f}")
    print(f"Sphere residual RMS: {np.sqrt(np.mean(result.fun**2)):.4f}")
    return cx, cy, cz, r

def fit_cylinder_3d(targets):
    cx0 = targets[:, 0].mean()
    cy0 = targets[:, 1].mean()
    cz0 = targets[:, 2].mean()
    r0  = np.mean(np.sqrt((targets[:, 0] - cx0)**2 +
                           (targets[:, 1] - cy0)**2))
    x0 = [cx0, cy0, cz0, 0, 0, 1, r0]
    result = least_squares(cylinder_residuals_3d, x0, args=(targets,))
    cx, cy, cz, ax, ay, az, r = result.x
    axis = np.array([ax, ay, az])
    axis = axis / np.linalg.norm(axis)
    print(f"Cylinder centre: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
    print(f"Cylinder axis:   ({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f})")
    print(f"Cylinder radius: {r:.3f}")
    print(f"Cylinder residual RMS: {np.sqrt(np.mean(result.fun**2)):.4f}")
    return cx, cy, cz, axis, r

# ---------------------------------------------------------------
# Load and normalise
# ---------------------------------------------------------------

coords = readObjVertices(filepath)
minimumZ = min(coords[:, 2])
coords[:, 2] = coords[:, 2] + minimumZ * -1
print(coords.shape)

obj_name = os.path.splitext(os.path.basename(filepath))[0]

# ---------------------------------------------------------------
# Sphere seed: furthest in [-x, -y, +z] or [+x, -y, +z] corner
# ---------------------------------------------------------------

if 'right' in obj_name.lower():
    sphere_direction = np.array([1, -1, 1], dtype=float)
else:
    sphere_direction = np.array([-1, -1, 1], dtype=float)

sphere_direction = sphere_direction / np.linalg.norm(sphere_direction)
projections  = coords @ sphere_direction
corner_index = int(np.argmax(projections))
maxz_index   = int(np.argmax(coords[:, 2]))

print(f"Corner node index: {corner_index}, coords: {coords[corner_index]}")
print(f"Max Z node index:  {maxz_index},  coords: {coords[maxz_index]}")

# ---------------------------------------------------------------
# Sphere fitting on femoral head
# ---------------------------------------------------------------

targets_indices = get_n_neighbours(coords, corner_index, N)
targets         = coords[targets_indices]
cx_s, cy_s, cz_s, r_s = fit_sphere(targets)

# ---------------------------------------------------------------
# Cylinder seed: lowest node above cutoff, shaft coords only
# ---------------------------------------------------------------

cutoff_z          = CYL_CUTOFF_MM
shaft_mask        = coords[:, 2] > cutoff_z
shaft_indices     = np.where(shaft_mask)[0]
shaft_only_coords = coords[shaft_mask]

cyl_seed_local = int(np.argmin(shaft_only_coords[:, 2]))
cyl_seed_index = int(shaft_indices[cyl_seed_local])

print(f"Cylinder seed node: {cyl_seed_index}, coords: {coords[cyl_seed_index]}")

# ---------------------------------------------------------------
# Cylinder fitting on shaft
# ---------------------------------------------------------------

cyl_targets_indices_local = get_n_neighbours(shaft_only_coords, cyl_seed_local, N_CYL)
cyl_targets               = shaft_only_coords[cyl_targets_indices_local]
cx_c, cy_c, cz_c, cyl_axis, r_c = fit_cylinder_3d(cyl_targets)

ignored_coords = coords[coords[:, 2] <= cutoff_z]

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------

sampled = coords[::5]

fig = plt.figure(figsize=(12, 10))
ax  = fig.add_subplot(111, projection='3d')

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

# Mesh nodes
ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2],
           c='steelblue', s=1, alpha=0.3, label='Mesh nodes')

# Ignored bottom nodes
ax.scatter(ignored_coords[::5, 0], ignored_coords[::5, 1], ignored_coords[::5, 2],
           c='grey', s=1, alpha=0.2, label=f'Ignored nodes (<{cutoff_z:.1f}mm)')

# Femoral head target points
ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2],
           c='red', s=20, zorder=5, label=f'Femoral head points (n={N})')

# Cylinder target points
ax.scatter(cyl_targets[:, 0], cyl_targets[:, 1], cyl_targets[:, 2],
           c='orange', s=20, zorder=5, label=f'Cylinder points (n={N_CYL})')

# Seed nodes
for idx in [corner_index, maxz_index, cyl_seed_index]:
    ax.scatter(coords[idx, 0], coords[idx, 1], coords[idx, 2],
               c='yellow', s=150, zorder=6, edgecolors='black')
ax.scatter([], [], [], c='yellow', edgecolors='black', s=150, label='Seed nodes')

# Fitted sphere
u  = np.linspace(0, 2 * np.pi, 30)
v  = np.linspace(0, np.pi, 30)
sx = cx_s + r_s * np.outer(np.cos(u), np.sin(v))
sy = cy_s + r_s * np.outer(np.sin(u), np.sin(v))
sz = cz_s + r_s * np.outer(np.ones_like(u), np.cos(v))
ax.plot_wireframe(sx, sy, sz, color='green', alpha=0.2, linewidth=0.5,
                  label=f'Fitted sphere R={r_s:.2f}mm')
ax.scatter(cx_s, cy_s, cz_s, c='green', s=100, zorder=5, label='Femoral head centre')

# Fitted cylinder - build surface along free axis
cyl_centre = np.array([cx_c, cy_c, cz_c])
cyl_length = cyl_targets[:, 2].max() - cyl_targets[:, 2].min()

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

ax.plot_wireframe(x_cyl, y_cyl, z_cyl,
                  color='orange', alpha=0.2, linewidth=0.5,
                  label=f'Fitted cylinder R={r_c:.2f}mm')

p1 = cyl_centre - cyl_axis * cyl_length / 2
p2 = cyl_centre + cyl_axis * cyl_length / 2
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
        c='orange', linewidth=2, label='Cylinder axis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(loc='upper left', fontsize=7)
plt.title(f'{obj_name} — Sphere R={r_s:.2f}mm | Cylinder R={r_c:.2f}mm')
plt.tight_layout()

os.makedirs('graphImages', exist_ok=True)
plt.savefig(f'graphImages/{obj_name}_fit.png', dpi=150, bbox_inches='tight')
plt.show()