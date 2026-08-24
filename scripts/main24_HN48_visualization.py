import os
import time
import warnings
import pickle

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import seaborn as sns
import trimesh
from scipy.interpolate import griddata
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

warnings.filterwarnings('ignore')

# ==========================================
# 0. 全局与期刊级绘图设置 (随意修改这里的参数，秒生效)
# ==========================================
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    'font.sans-serif': ['Arial', 'Helvetica', 'SimHei'],
    'axes.unicode_minus': False,
    'figure.dpi': 900,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.5,
    'font.size': 24,
    'axes.labelsize': 27,
    'axes.titlesize': 30,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})

COLOR_PALETTE = {
    'source': '#00A087', 'sink': '#E64B35', 'fit': '#3C5488',
    'truth_dot': '#DC0000', 'truth_line': '#4DBBD5', 'scatter': '#B0B0B0',
    'surface': 'viridis', 'heatmap': 'RdYlBu_r'
}

# 需要用到的基础常数字典（为了脚本独立运行复制过来）
PAPER_DATA = {
    'DN251': {'r': np.array([50.0, 50.0, 50.0, 50.0, 50.0]), 'p': np.array([7.0, 8.0, 9.0, 10.0, 11.0]),
              'y': np.array([144.444, 171.556, 182.213, 162.222, 187.556])},
    'DN252': {'r': np.array([50.0, 50.0, 50.0, 50.0, 50.0]), 'p': np.array([7.0, 8.0, 9.0, 10.0, 11.0]),
              'y': np.array([173.333, 153.333, 190.186, 173.333, 189.333])},
    'DN253': {'r': np.array([50.0, 50.0, 50.0, 50.0, 50.0]), 'p': np.array([7.0, 8.0, 9.0, 10.0, 11.0]),
              'y': np.array([193.778, 179.111, 157.778, 166.667, 169.333])},
    'HN48': {'r': np.array([50.0, 50.0, 50.0, 50.0, 50.0]), 'p': np.array([7.0, 8.0, 9.0, 10.0, 11.0]),
             'y': np.array([157.778, 183.111, 175.024, 179.556, 182.222])},
    'HN51': {'r': np.array([50.0, 50.0, 50.0, 50.0, 50.0]), 'p': np.array([7.0, 8.0, 9.0, 10.0, 11.0]),
             'y': np.array([185.333, 190.222, 177.112, 150.222, 148.444])}
}

TRUTH_DENSITY = {"DN253": 18.0, "DN252": 18.0, "DN251": 22.0, "HN48": 28.0, "HN51": 25.0}
BASE_DIR = r"Kaggle数据"
YEAR_SELECT = "2019"
FOUR_D_CONFIG = {
    "variety": "HN48",
    "dates": ["0529", "0603", "0608", "0612", "0618", "0624", "0627", "0705", "0826", "0921"],
    "flip_axis": "X"
}
MEASURED_ROW_SPACING_CM = 50.0
MEASURED_PLANT_SPACINGS_CM = np.array([7.0, 8.0, 9.0, 10.0, 11.0])
MESH_SIMPLIFY_FACES = 30000


# 读取 3D 模型的辅助函数（渲染3D图必须）
def load_mesh(file_path, flip_axis="X"):
    mesh_o3d = o3d.io.read_triangle_mesh(file_path)
    if len(mesh_o3d.triangles) > MESH_SIMPLIFY_FACES:
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(MESH_SIMPLIFY_FACES)
    mesh = trimesh.Trimesh(vertices=np.asarray(mesh_o3d.vertices), faces=np.asarray(mesh_o3d.triangles))
    if flip_axis == "X":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    elif flip_axis == "Y":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif flip_axis == "-X":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    mesh.apply_translation([-mesh.centroid[0], -mesh.centroid[1], -mesh.bounds[0][2]])
    return mesh




def select_best_on_fixed_row(rf_model, row_spacing=MEASURED_ROW_SPACING_CM,
                             p_min=4.0, p_max=30.0, n=600):
    """
    在固定 50 cm 行距切片上搜索 Yield_Score 最高的株距。
    这是二维响应面中的 measured transect / fixed-row transect 分析。
    """
    p_grid = np.linspace(p_min, p_max, n)
    grid = pd.DataFrame({
        'Row_Spacing': np.full_like(p_grid, row_spacing, dtype=float),
        'Plant_Spacing': p_grid
    })
    score = rf_model.predict(grid)
    idx = int(np.argmax(score))
    return {
        'r': float(row_spacing),
        'p': float(p_grid[idx]),
        'density': float(10000.0 / (row_spacing * p_grid[idx])),
        'score': float(score[idx]),
        'p_grid': p_grid,
        'score_grid': score
    }


def get_measured_transect_points(variety):
    """返回固定 50 cm 行距下的 5 个实测株距点。"""
    r_emp = PAPER_DATA[variety]['r'].astype(float)
    p_emp = PAPER_DATA[variety]['p'].astype(float)
    y_emp = PAPER_DATA[variety]['y'].astype(float)
    d_emp = 10000.0 / (r_emp * p_emp)
    return pd.DataFrame({
        'Row_Spacing': r_emp,
        'Plant_Spacing': p_emp,
        'Density': d_emp,
        'True_Yield': y_emp
    })

# ==========================================
# 5. 可视化函数群 (保留原代码的所有图表逻辑)
# ==========================================
def draw_yield_density_curve(df, best_d, variety):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=df, x='Density', y='Yield_Score', color=COLOR_PALETTE['scatter'], alpha=0.6, s=60, ax=ax,
                    edgecolor='none')
    z = np.polyfit(df['Density'], df['Yield_Score'], 3)
    p_poly = np.poly1d(z)
    xr = np.linspace(df['Density'].min(), df['Density'].max(), 100)
    ax.plot(xr, p_poly(xr), color=COLOR_PALETTE['fit'], lw=4, label='Whole-Season Yield Response')
    rho_truth = TRUTH_DENSITY[variety]
    ax.axvline(best_d, color=COLOR_PALETTE['fit'], ls='-', lw=3, label=f"4D Sim Optima: {best_d:.1f}")
    ax.axvline(rho_truth, color=COLOR_PALETTE['truth_line'], ls='--', lw=3, label=f"Field Truth: {rho_truth:.1f}")
    ax.set_xlabel('Planting Density (plants/m²)')
    ax.set_ylabel('Cumulative Yield Score')
    ax.set_title("Yield-Density Response Manifold")
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02))
    sns.despine()
    plt.savefig(f"{variety}_Fig01_Yield_Density_Curve.png")
    plt.close()


def draw_dynamic_trajectory(df, ts_data, best_d, variety):
    fig, ax = plt.subplots(figsize=(10, 8))
    df_sorted = df.sort_values(by='Density').reset_index(drop=True)
    idx_list = [int(len(df_sorted) * x) for x in [0.10, 0.25, 0.50, 0.90]]
    styles = [('#A0A0A0', 'o', 'Low'), ('#707070', 's', 'Q1'), ('#404040', '^', 'Med'),
              (COLOR_PALETTE['sink'], 'v', 'High')]
    doys = [d['doy'] for d in ts_data]
    for idx, (color, marker, label) in zip(idx_list, styles):
        ax.plot(doys, df_sorted.iloc[idx]['LIE_Trajectory'], marker=marker, color=color, lw=2.5, markersize=8,
                alpha=0.8,
                label=f"{label} Den ({df_sorted.iloc[idx]['Density']:.1f})")
    idx_opt = (df_sorted['Density'] - best_d).abs().idxmin()
    ax.plot(doys, df_sorted.iloc[idx_opt]['LIE_Trajectory'], marker='D', markersize=10, color=COLOR_PALETTE['source'],
            lw=4, zorder=10, label=f"Optimal Target ({best_d:.1f})")
    ax.set_xlabel('Day of Year (DOY)')
    ax.set_ylabel('Light Interception Efficiency (LIE)')
    ax.set_title("Dynamic 4D Canopy Closure Trajectory")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', frameon=True)
    sns.despine()
    plt.savefig(f"{variety}_Fig02_LIE_Trajectory.png")
    plt.close()


def draw_density_cum_lie_curve(df, best_d, variety):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=df, x='Density', y='Cum_LIE', color=COLOR_PALETTE['scatter'], alpha=0.6, s=60, ax=ax,
                    edgecolor='none')
    z = np.polyfit(df['Density'], df['Cum_LIE'], 3)
    p_poly = np.poly1d(z)
    xr = np.linspace(df['Density'].min(), df['Density'].max(), 100)
    ax.plot(xr, p_poly(xr), color=COLOR_PALETTE['source'], lw=4, label='Cumulative LIE Trend')
    ax.axvline(best_d, color=COLOR_PALETTE['fit'], ls='--', lw=3, label=f"Optimal Density: {best_d:.1f}")
    ax.set_xlabel('Planting Density (plants/m²)')
    ax.set_ylabel('Full-Season Cum_LIE')
    ax.set_title("Density vs. Cumulative Light Interception Efficiency")
    ax.legend(loc='lower right')
    sns.despine()
    plt.savefig(f"{variety}_Fig11_Density_vs_CumLIE.png")
    plt.close()


def draw_3d_optical_absorption(base_mesh, variety, date_str):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    vertices, faces = np.asarray(base_mesh.vertices), np.asarray(base_mesh.faces)
    z_max, z_min = vertices[:, 2].max(), vertices[:, 2].min()
    depth = np.clip((z_max - vertices[faces].mean(axis=1)[:, 2]) / (z_max - z_min + 1e-5), 0, 1)
    colors = plt.cm.get_cmap('YlGn')(1.0 - np.exp(-1.8 * depth))
    ax.add_collection3d(Poly3DCollection(vertices[faces], facecolors=colors, edgecolors='none'))
    ax.set_xlim([vertices[:, 0].min(), vertices[:, 0].max()])
    ax.set_ylim([vertices[:, 1].min(), vertices[:, 1].max()])
    ax.set_zlim([z_min, z_max])
    ax.axis('off')
    ax.set_title(f"3D Optical Absorption Index ({date_str})")
    plt.savefig(f"{variety}_Fig03a_3D_Absorption_{date_str}.png", transparent=True)
    plt.close()


def draw_combined_vertical_penetration_profile(variety, profiles_data):
    fig, ax = plt.subplots(figsize=(8, 10))
    colors = sns.color_palette("husl", len(profiles_data))
    all_z_min, all_z_max = float('inf'), float('-inf')

    for i, (date_str, data) in enumerate(profiles_data.items()):
        vertices = np.asarray(data['mesh'].vertices)
        stats = data['stats']
        z_max, z_min = vertices[:, 2].max(), vertices[:, 2].min()
        all_z_min, all_z_max = min(all_z_min, z_min), max(all_z_max, z_max)
        h_bins = np.linspace(z_min, z_max, 30)
        h_centers = (h_bins[:-1] + h_bins[1:]) / 2
        trans = np.exp(-stats['k_ext_est'] * (
                    np.cumsum(np.histogram(vertices[:, 2], bins=h_bins)[0][::-1]) / len(vertices) * 5)) * 100
        ax.plot(trans, h_centers[::-1], lw=4, marker='o', markersize=8, label=date_str, color=colors[i])
        ax.fill_betweenx(h_centers[::-1], 0, trans, alpha=0.1, color=colors[i])

    ax.set_xlabel('Relative Transmittance (%)')
    ax.set_ylabel('Canopy Height (mm)')
    ax.set_xlim(0, 105)
    ax.set_ylim(all_z_min, all_z_max)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title("Vertical Light Penetration (All Stages)")
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    sns.despine()
    plt.savefig(f"{variety}_Fig03b_Vertical_Penetration.png", bbox_inches='tight')
    plt.close()


def draw_yield_surface_3d(df, variety, rf_r2):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(df['Row_Spacing'], df['Plant_Spacing'], df['Yield_Score'], cmap=COLOR_PALETTE['surface'],
                           alpha=0.9, edgecolor='none')
    ax.view_init(25, -60)
    ax.set_xlabel('\nRow Spacing (cm)', linespacing=2)
    ax.set_ylabel('\nPlant Spacing (cm)', linespacing=2)
    ax.set_zlabel('\nCum Yield Score', linespacing=2)
    ax.set_title("Cumulative Yield Surface")
    ax.text2D(0.05, 0.95, f"RF Model $R^2$: {rf_r2:.3f}", transform=ax.transAxes,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.savefig(f"{variety}_Fig04_Yield_Surface_3D.png")
    plt.close()


def draw_spatial_heatmap(df, grids, best_config, variety, rf_model=None, fixed50_config=None):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    Xi, Yi = grids
    Zi = griddata((df['Row_Spacing'], df['Plant_Spacing']), df['Yield_Score'], (Xi, Yi), method='cubic')
    c = ax.contourf(Xi, Yi, Zi, levels=40, cmap=COLOR_PALETTE['heatmap'])
    cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Digital-twin Yield_Score", fontweight='bold')

    # 二维响应面候选点
    ax.scatter(best_config['r'], best_config['p'], c='white', marker='*', s=430,
               edgecolors='black', linewidths=1.6, label='Virtual 2D candidate', zorder=12)

    # 50 cm 实测切片：虚线 + 5 个实测处理点
    p_min, p_max = ax.get_ylim()
    ax.axvline(MEASURED_ROW_SPACING_CM, color='black', linestyle='--', linewidth=2.2,
               label='Measured 50 cm row transect', zorder=10)
    ax.scatter(np.full_like(MEASURED_PLANT_SPACINGS_CM, MEASURED_ROW_SPACING_CM),
               MEASURED_PLANT_SPACINGS_CM,
               c='black', marker='o', s=110, edgecolors='white', linewidths=1.0,
               label='Field-tested spacings', zorder=13)

    # 固定 50 cm 行距切片上的模型候选株距
    if fixed50_config is None and rf_model is not None:
        fixed50_config = select_best_on_fixed_row(rf_model)
    if fixed50_config is not None:
        ax.scatter(fixed50_config['r'], fixed50_config['p'], c=COLOR_PALETTE['truth_dot'],
                   marker='D', s=180, edgecolors='white', linewidths=1.2,
                   label=f"Best on 50 cm transect ({fixed50_config['p']:.1f} cm)", zorder=14)

    ax.set_xlabel('Row spacing (cm)')
    ax.set_ylabel('Plant spacing (cm)')
    ax.set_title("2D response surface with 50 cm measured transect")
    ax.legend(loc='lower right', frameon=True)
    sns.despine()
    plt.savefig(f"{variety}_Fig05_2D_Response_Surface_With_50cm_Transect.png")
    plt.close()


def draw_canopy_3d(variety, r, p, base_mesh, date_str):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(base_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(base_mesh.faces)
    if len(o3d_mesh.triangles) > 5000: o3d_mesh = o3d_mesh.simplify_quadric_decimation(5000)
    o3d_mesh.compute_triangle_normals()
    normals, vis_faces = np.asarray(o3d_mesh.triangle_normals), np.asarray(o3d_mesh.triangles)
    base_vertices = np.asarray(o3d_mesh.vertices)

    min_dist_truth = 10000.0 / (50.0 * TRUTH_DENSITY[variety])
    min_dist_current = min(r, p)
    sas_ratio_spatial = min_dist_truth / min_dist_current
    sas_z = np.clip(1.0 + 0.15 * np.log(sas_ratio_spatial + 1e-5), 0.8, 1.4)
    sas_xy = np.clip(1.0 - 0.08 * np.log(sas_ratio_spatial + 1e-5), 0.75, 1.15)

    z_max = base_vertices[:, 2].max() * 0.1 * sas_z
    ground_z = 0
    light_dir = np.array([0.7, 0.4, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    intensities = 0.35 + (1 - 0.35) * np.clip(np.dot(normals, light_dir), 0, 1)
    base_leaf_color = np.array([0.25, 0.55, 0.22])

    all_triangles, all_colors, shadow_triangles = [], [], []
    ox, oy = -r / 2.0, -p / 2.0

    for row in range(2):
        for col in range(2):
            local_shrink = 0.1 * np.random.uniform(0.95, 1.05)
            shifted_vertices = base_vertices.copy()
            shifted_vertices[:, 0] *= (local_shrink * sas_xy)
            shifted_vertices[:, 1] *= (local_shrink * sas_xy)
            shifted_vertices[:, 2] *= (local_shrink * sas_z)

            theta = np.random.uniform(0, 2 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            x_rot = shifted_vertices[:, 0] * cos_t - shifted_vertices[:, 1] * sin_t
            y_rot = shifted_vertices[:, 0] * sin_t + shifted_vertices[:, 1] * cos_t

            shifted_vertices[:, 0] = x_rot + (ox + row * r)
            shifted_vertices[:, 1] = y_rot + (oy + col * p)
            shifted_vertices[:, 2] = np.maximum(shifted_vertices[:, 2], ground_z)

            triangles = shifted_vertices[vis_faces]
            all_triangles.extend(triangles)

            shaded_colors = np.zeros((len(triangles), 4))
            shaded_colors[:, :3] = base_leaf_color * intensities[:, np.newaxis]
            shaded_colors[:, 3] = 0.98
            all_colors.extend(shaded_colors)

            shadow_verts = shifted_vertices.copy()
            h = shadow_verts[:, 2] - ground_z
            shadow_verts[:, 0] -= h * (light_dir[0] / light_dir[2])
            shadow_verts[:, 1] -= h * (light_dir[1] / light_dir[2])
            shadow_verts[:, 2] = ground_z - 0.5
            shadow_triangles.extend(shadow_verts[vis_faces])

    if hasattr(ax, 'computed_zorder'): ax.computed_zorder = False
    field_w, field_h = r * 2.5, p * 2.5
    ground_z_surface = ground_z - 1.0
    x_edges, y_edges = np.linspace(-field_w / 2, field_w / 2, 25), np.linspace(-field_h / 2, field_h / 2, 25)

    ground_faces = []
    for i in range(len(x_edges) - 1):
        for j in range(len(y_edges) - 1):
            x0, x1 = x_edges[i], x_edges[i + 1]
            y0, y1 = y_edges[j], y_edges[j + 1]
            ground_faces.append([(x0, y0, ground_z_surface), (x1, y0, ground_z_surface), (x1, y1, ground_z_surface),
                                 (x0, y1, ground_z_surface)])

    ground_poly = Poly3DCollection(ground_faces, facecolors='#4A3320', edgecolors='#2A1A10', linewidths=0.5, alpha=1.0,
                                   zorder=0)
    ax.add_collection3d(ground_poly)
    ax.add_collection3d(
        Poly3DCollection(shadow_triangles, facecolors='#18120E', edgecolors='none', alpha=0.35, zorder=1))
    ax.add_collection3d(Poly3DCollection(all_triangles, facecolors=all_colors, edgecolors='none', zorder=2))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_axis_off()
    ax.set_box_aspect((1, field_h / field_w, z_max / field_w * 1.8))
    ax.view_init(elev=20, azim=-50)
    ax.set_xlim([-field_w / 2, field_w / 2])
    ax.set_ylim([-field_h / 2, field_h / 2])
    ax.set_zlim([ground_z - 2, z_max * 1.15])
    plt.title(f"High-Fidelity Canopy (SAS Manifested) - {date_str}", y=0.95)
    plt.savefig(f"{variety}_Fig06_Canopy_3D_{date_str}.png", transparent=True)
    plt.close()


def draw_source_sink_balance(df, variety):
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax2 = ax1.twinx()
    sns.regplot(data=df, x='Density', y='LIE', ax=ax1, scatter=False, order=3, color=COLOR_PALETTE['source'],
                label='Seasonal Avg Source (LIE)')
    sns.regplot(data=df, x='Density', y='HI', ax=ax2, scatter=False, order=3, color=COLOR_PALETTE['sink'],
                label='Peak Sink (HI)')
    ax1.set_xlabel('Planting Density (plants/m²)')
    ax1.set_ylabel('Seasonal Avg LIE', color=COLOR_PALETTE['source'], fontweight='bold')
    ax2.set_ylabel('Harvest Index (HI)', color=COLOR_PALETTE['sink'], fontweight='bold')
    ax1.set_title("Spatiotemporal Source-Sink Balance")
    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(0, 0.65)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', frameon=False)
    sns.despine(right=False)
    plt.savefig(f"{variety}_Fig07_Source_Sink_Balance.png")
    plt.close()


def draw_model_truth_alignment(df, best_d, variety):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=df, x='Density', y='Yield_Score', color=COLOR_PALETTE['scatter'], alpha=0.5, s=50, ax=ax,
                    edgecolor='none')
    z = np.polyfit(df['Density'], df['Yield_Score'], 3)
    p_poly = np.poly1d(z)
    xr = np.linspace(df['Density'].min(), df['Density'].max(), 100)
    ax.plot(xr, p_poly(xr), color=COLOR_PALETTE['fit'], lw=4, label='Cum Model Trend')
    rho_truth, truth_yield_est = TRUTH_DENSITY[variety], p_poly(TRUTH_DENSITY[variety])
    ax.axvline(best_d, color=COLOR_PALETTE['fit'], ls='-', lw=2.5, label=f"4D Opt: {best_d:.1f}")
    ax.axvline(rho_truth, color=COLOR_PALETTE['truth_line'], ls='--', lw=2.5)
    ax.scatter([rho_truth], [truth_yield_est], color=COLOR_PALETTE['truth_dot'], s=300, marker='o', zorder=10,
               edgecolors='white', linewidths=2.5, label=f'Field Truth: {rho_truth:.1f}')
    ax.set_xlabel('Planting Density (plants/m²)')
    ax.set_ylabel('Cumulative Yield Score')
    ax.set_title("4D Model-Truth Alignment")
    ax.legend(frameon=True, loc='lower center', bbox_to_anchor=(0.5, 0.05))
    sns.despine()
    plt.savefig(f"{variety}_Fig08_Model_Truth_Alignment.png")
    plt.close()


def draw_absolute_yield_mechanistic(df, best_config, variety, rf_model, fixed50_config=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    r_emp, p_emp, y_emp = PAPER_DATA[variety]['r'], PAPER_DATA[variety]['p'], PAPER_DATA[variety]['y']
    emp_densities = 10000.0 / (r_emp * p_emp)

    grid_emp = pd.DataFrame({'Row_Spacing': r_emp, 'Plant_Spacing': p_emp})
    y_score_emp = rf_model.predict(grid_emp)

    def power_law_model(score, k_scale, gamma):
        return k_scale * (score ** gamma)

    loocv_errors = []
    for i in range(len(y_emp)):
        train_idx = [j for j in range(len(y_emp)) if j != i]
        score_train, y_train = y_score_emp[train_idx], y_emp[train_idx]
        try:
            popt_cv, _ = curve_fit(power_law_model, score_train, y_train,
                                   p0=[np.mean(y_train) / np.mean(score_train), 1.0], bounds=([0, 0.1], [np.inf, 2.5]))
        except RuntimeError:
            popt_cv = [np.sum(score_train * y_train) / np.sum(score_train ** 2), 1.0]
        y_pred = power_law_model(y_score_emp[i], popt_cv[0], popt_cv[1])
        loocv_errors.append(abs(y_pred - y_emp[i]) / y_emp[i])
    mean_mape = np.mean(loocv_errors) * 100

    try:
        popt_final, _ = curve_fit(power_law_model, y_score_emp, y_emp, p0=[np.mean(y_emp) / np.mean(y_score_emp), 1.0],
                                  bounds=([0, 0.1], [np.inf, 2.5]))
        k_opt, gamma_opt = popt_final[0], popt_final[1]
    except RuntimeError:
        k_opt, gamma_opt = np.sum(y_score_emp * y_emp) / np.sum(y_score_emp ** 2), 1.0

    # 固定 50 cm 行距切片曲线：这里把已有实测处理和连续株距候选联系起来
    r_sim, p_sim = MEASURED_ROW_SPACING_CM, np.linspace(4, 30, 160)
    densities_sim = 10000.0 / (r_sim * p_sim)
    grid_sim = pd.DataFrame({'Row_Spacing': np.full_like(p_sim, r_sim), 'Plant_Spacing': p_sim})
    score_sim = rf_model.predict(grid_sim)
    yield_sim = power_law_model(score_sim, k_opt, gamma_opt)

    sort_idx = np.argsort(densities_sim)
    ax.plot(densities_sim[sort_idx], yield_sim[sort_idx], color='#1f77b4', lw=4,
            label=f'50 cm transect calibration ($\gamma$={gamma_opt:.2f})')
    ax.scatter(emp_densities, y_emp, color='black', marker='o', s=100, zorder=5, label='Field treatments at 50 cm')

    # 50 cm 行距切片上的模型候选株距
    if fixed50_config is None:
        fixed50_config = select_best_on_fixed_row(rf_model)
    fixed50_score = rf_model.predict(pd.DataFrame({'Row_Spacing': [fixed50_config['r']],
                                                   'Plant_Spacing': [fixed50_config['p']]}))[0]
    fixed50_y_abs = power_law_model(fixed50_score, k_opt, gamma_opt)
    ax.scatter([fixed50_config['density']], [fixed50_y_abs], color=COLOR_PALETTE['source'], s=280,
               marker='D', zorder=10, edgecolors='white', linewidths=1.2,
               label=f"Best on 50 cm transect: {fixed50_config['p']:.1f} cm")
    ax.axvline(fixed50_config['density'], color=COLOR_PALETTE['source'], ls='--', lw=2.3)

    # 二维响应面候选点：仅作为虚拟二维候选，与 50 cm 切片候选区分
    best_score = rf_model.predict(pd.DataFrame({'Row_Spacing': [best_config['r']], 'Plant_Spacing': [best_config['p']]}))[0]
    best_y_abs = power_law_model(best_score, k_opt, gamma_opt)
    ax.scatter([best_config['density']], [best_y_abs], color=COLOR_PALETTE['truth_dot'], s=360,
               marker='*', zorder=11, edgecolors='white', linewidths=1.3,
               label=f"Virtual 2D candidate: {best_config['r']:.1f}×{best_config['p']:.1f} cm")

    ax.text(1.05, 0.62,
            f"50 cm transect back-testing:\nLOOCV MAPE: {mean_mape:.1f}%\n$K_{{scale}}$: {k_opt:.1f}\n$\gamma$: {gamma_opt:.2f}",
            transform=ax.transAxes, fontsize=21, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlabel('Planting density (plants/m²)')
    ax.set_ylabel('Yield-scale reference (kg/Mu)')
    ax.set_title("50 cm row-spacing transect yield-scale calibration")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    sns.despine()
    plt.savefig(f"{variety}_Fig09_50cm_Transect_Yield_Scale_Calibration.png", bbox_inches='tight')
    return best_y_abs


def draw_validation_dumbbell(variety, best_d, truth_d):
    fig, ax = plt.subplots(figsize=(12, 4))
    err_pct = abs(best_d - truth_d) / truth_d * 100
    ax.plot([truth_d, best_d], [0, 0], color='#8b8c89', linewidth=4, zorder=1, alpha=0.8)
    ax.scatter(truth_d, 0, color=COLOR_PALETTE['truth_line'], s=400, edgecolor='white', linewidth=2, zorder=2,
               label='Field Truth')
    ax.scatter(best_d, 0, color=COLOR_PALETTE['truth_dot'], s=400, edgecolor='white', linewidth=2, zorder=2,
               label='4D Sim Optima')
    ax.text((truth_d + best_d) / 2, 0.08, f"$\Delta$ {err_pct:.1f}%", ha='center', va='bottom', fontsize=24,
            fontweight='bold')
    ax.text(truth_d, -0.08, f"{truth_d:.1f}", ha='center', va='top', fontsize=21, color='gray', fontweight='bold')
    ax.text(best_d, -0.08, f"{best_d:.1f}", ha='center', va='top', fontsize=21, color='gray', fontweight='bold')
    ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xticks([])
    ax.set_title(f"Agronomic Digital Twin Validation", fontsize=30, fontweight='bold', pad=20)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{variety}_Fig10_Validation_Dumbbell.png")
    plt.close()
    return err_pct


# ==========================================
# 6. 主程序 - 只执行轻量拟合和渲染出图
# ==========================================


def draw_fixed50_transect_yieldscore(df, variety, rf_model, fixed50_config):
    """
    新增图：固定行距 50 cm 下，连续株距 Yield_Score 曲线 + 5 个实测处理位置。
    用于把“二维响应面”与“固定行距株距测试”连接起来。
    """
    fig, ax1 = plt.subplots(figsize=(10.5, 8))

    p_grid = fixed50_config['p_grid']
    score_grid = fixed50_config['score_grid']
    ax1.plot(p_grid, score_grid, color=COLOR_PALETTE['fit'], lw=4,
             label='RF-smoothed Yield_Score on 50 cm transect')
    ax1.scatter([fixed50_config['p']], [fixed50_config['score']], color=COLOR_PALETTE['source'],
                marker='D', s=250, edgecolors='white', linewidths=1.2,
                label=f"Model-selected spacing: {fixed50_config['p']:.1f} cm", zorder=10)

    emp = get_measured_transect_points(variety)
    emp_grid = pd.DataFrame({'Row_Spacing': emp['Row_Spacing'], 'Plant_Spacing': emp['Plant_Spacing']})
    emp_score = rf_model.predict(emp_grid)
    ax1.scatter(emp['Plant_Spacing'], emp_score, color='black', marker='o', s=110,
                edgecolors='white', linewidths=1.0, label='Field-tested spacings', zorder=9)

    ax2 = ax1.twinx()
    ax2.scatter(emp['Plant_Spacing'], emp['True_Yield'], color=COLOR_PALETTE['truth_dot'], marker='^', s=110,
                edgecolors='white', linewidths=1.0, label='Observed yield', zorder=8)

    observed_best = emp.loc[emp['True_Yield'].idxmax()]
    ax2.axvline(observed_best['Plant_Spacing'], color=COLOR_PALETTE['truth_dot'], linestyle=':', lw=2.5,
                label=f"Observed-best spacing: {observed_best['Plant_Spacing']:.0f} cm")
    ax1.axvline(fixed50_config['p'], color=COLOR_PALETTE['source'], linestyle='--', lw=2.5)

    ax1.set_xlabel('Plant spacing under fixed 50 cm row spacing (cm)')
    ax1.set_ylabel('Digital-twin Yield_Score')
    ax2.set_ylabel('Observed yield (kg/Mu)')
    ax1.set_title('Fixed 50 cm row-spacing transect: plant-spacing response')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='best', frameon=True)
    ax1.grid(alpha=0.25)
    sns.despine(ax=ax1, right=False)
    plt.savefig(f"{variety}_Fig10_Fixed_50cm_Transect_Plant_Spacing_Response.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    variety = FOUR_D_CONFIG['variety']
    data_file = f"{variety}_sim_data.pkl"

    print(f"\n=============================================")
    print(f"  🎨 启动可视化引擎 (热重载模式) ")
    print(f"=============================================\n")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"未找到 {data_file}！请先运行 step1_simulation.py 生成基础数据。")

    print(f"[{time.strftime('%H:%M:%S')}] 加载仿真数据缓存...")
    with open(data_file, "rb") as f:
        cache = pickle.load(f)

    df_results = cache["df_results"]
    doys = cache["doys"]
    ts_data = cache["ts_data"]
    pheno_records = cache["pheno_records"]

    print(f"[{time.strftime('%H:%M:%S')}] 快速拟合响应曲面 (毫秒级)...")
    X, y_score = df_results[['Row_Spacing', 'Plant_Spacing']], df_results['Yield_Score']
    rf = RandomForestRegressor(n_estimators=300, max_depth=15).fit(X, y_score)
    Xi, Yi = np.meshgrid(np.linspace(15, 75, 100), np.linspace(4, 30, 100))
    grid_df = pd.DataFrame({'Row_Spacing': Xi.ravel(), 'Plant_Spacing': Yi.ravel()})
    preds = rf.predict(grid_df)
    max_idx = np.argmax(preds)
    best_config = {'r': grid_df.iloc[max_idx]['Row_Spacing'], 'p': grid_df.iloc[max_idx]['Plant_Spacing'],
                   'density': 10000 / (grid_df.iloc[max_idx]['Row_Spacing'] * grid_df.iloc[max_idx]['Plant_Spacing'])}
    best_d = best_config['density']
    r2 = rf.score(X, y_score)

    # 50 cm 固定行距切片候选；用于把实测切片和二维响应面连接起来
    fixed50_config = select_best_on_fixed_row(rf)
    summary_df = pd.DataFrame([
        {
            'Variety': variety,
            'Candidate_Type': 'Virtual_2D_response_surface',
            'Row_Spacing_cm': best_config['r'],
            'Plant_Spacing_cm': best_config['p'],
            'Density_plants_m2': best_config['density'],
            'RF_Surrogate_R2_against_YieldScore': r2
        },
        {
            'Variety': variety,
            'Candidate_Type': 'Best_on_measured_50cm_transect',
            'Row_Spacing_cm': fixed50_config['r'],
            'Plant_Spacing_cm': fixed50_config['p'],
            'Density_plants_m2': fixed50_config['density'],
            'RF_Surrogate_R2_against_YieldScore': r2
        }
    ])
    summary_df.to_csv(f"{variety}_candidate_summary_2D_and_50cm_transect.csv", index=False, encoding='utf-8-sig')
    get_measured_transect_points(variety).to_csv(f"{variety}_measured_50cm_transect_points.csv", index=False, encoding='utf-8-sig')

    print(f"[{time.strftime('%H:%M:%S')}] 开始批量渲染 2D 统计图...")
    draw_yield_density_curve(df_results, best_d, variety)
    draw_dynamic_trajectory(df_results, ts_data, best_d, variety)
    draw_density_cum_lie_curve(df_results, best_d, variety)
    draw_yield_surface_3d(df_results, variety, r2)
    fixed50_config = select_best_on_fixed_row(rf)
    draw_spatial_heatmap(df_results, (Xi, Yi), best_config, variety, rf_model=rf, fixed50_config=fixed50_config)
    draw_source_sink_balance(df_results, variety)
    draw_model_truth_alignment(df_results, best_d, variety)

    best_yield = draw_absolute_yield_mechanistic(df_results, best_config, variety, rf, fixed50_config=fixed50_config)
    draw_fixed50_transect_yieldscore(df_results, variety, rf, fixed50_config)
    err_pct = draw_validation_dumbbell(variety, best_d, TRUTH_DENSITY[variety])

    print(f"[{time.strftime('%H:%M:%S')}] 开始生成 3D 冠层演化与透光图 (需读取 .ply 文件)...")
    profiles_data = {}
    for date_str in FOUR_D_CONFIG['dates']:
        mesh_path = os.path.join(BASE_DIR, YEAR_SELECT, f"{variety}_{date_str}.ply")
        if not os.path.exists(mesh_path):
            mesh_path = os.path.join(BASE_DIR, f"{variety}_{date_str}.ply")

        if os.path.exists(mesh_path) and date_str in pheno_records:
            print(f"  └─ 渲染 3D 视角: {date_str}...")
            base_mesh = load_mesh(mesh_path, flip_axis=FOUR_D_CONFIG.get('flip_axis', 'X'))
            draw_3d_optical_absorption(base_mesh, variety, date_str)
            profiles_data[date_str] = {'mesh': base_mesh, 'stats': pheno_records[date_str]}
            draw_canopy_3d(variety, best_config['r'], best_config['p'], base_mesh, date_str)

    if profiles_data:
        draw_combined_vertical_penetration_profile(variety, profiles_data)

    print(f"\n✅ [{time.strftime('%H:%M:%S')}] 所有图表重新绘制完毕！")