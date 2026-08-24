import os
import time
import math
import datetime
import warnings
import pickle  # 用于持久化保存复杂数据结构

import numpy as np
import open3d as o3d
import pandas as pd
import trimesh
from scipy.optimize import curve_fit
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==========================================
# 1. 核心大田经验数据 & 参数获取
# ==========================================
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


def get_empirical_params(variety_name):
    if variety_name not in PAPER_DATA: variety_name = 'DN251'
    r_data, p_data, y_data = PAPER_DATA[variety_name]['r'], PAPER_DATA[variety_name]['p'], PAPER_DATA[variety_name]['y']
    x_data = 10000.0 / (r_data * p_data)

    def base_yield_model(x, a, b): return a * x * np.exp(-b * x)

    popt, _ = curve_fit(base_yield_model, x_data, y_data, p0=[30.0, 0.05], maxfev=10000)
    return popt[0], popt[1]


# ==========================================
# 2. 4D 全生育期时空实验参数
# ==========================================
BASE_DIR = r"Kaggle数据"
YEAR_SELECT = "2019"
NEAU_LAT, NEAU_LON = 45.74, 126.63
TIME_STEPS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

FIELD_SIZE, CORE_ZONE = (300, 300), (150, 150)
RAY_COUNT_PER_STEP = 3000
NUM_CONFIGS = 300
# 固定行距实测切片：二维响应面中的 50 cm row-spacing transect
MEASURED_ROW_SPACING_CM = 50.0
MEASURED_PLANT_SPACINGS_CM = np.array([7.0, 8.0, 9.0, 10.0, 11.0])
MESH_SIMPLIFY_FACES = 30000

RUE_CONST = 180.0
HI_MAX = 0.55

TRUTH_DENSITY = {"DN253": 18.0, "DN252": 18.0, "DN251": 22.0, "HN48": 28.0, "HN51": 25.0}
FOUR_D_CONFIG = {
    "variety": "DN251",
    "dates": ["0529", "0603", "0608", "0612", "0618", "0624", "0627", "0705", "0826", "0921"],
    "flip_axis": "X"
}


# ==========================================
# 3. 基础配置与表型分析
# ==========================================
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


def analyze_plant_phenotype(mesh):
    extents = mesh.extents
    avg_width = (extents[0] + extents[1]) / 2.0
    aspect_ratio = extents[2] / avg_width if avg_width > 0 else 1.0

    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    face_centers = vertices[faces].mean(axis=1)
    face_areas = mesh.area_faces

    z_max, z_min = vertices[:, 2].max(), vertices[:, 2].min()
    bins = np.linspace(z_min, z_max, 21)
    areas_per_bin = [face_areas[(face_centers[:, 2] >= bins[i]) & (face_centers[:, 2] < bins[i + 1])].sum() for i in
                     range(20)]

    cum_area = np.cumsum(np.array(areas_per_bin)[::-1])
    top_half_ratio = cum_area[len(cum_area) // 2] / (cum_area[-1] + 1e-5) if len(cum_area) > 0 else 1.0
    k_ext_est = 0.4 + (0.85 - 0.4) * top_half_ratio

    return {"raw_height": extents[2], "raw_width": avg_width, "aspect_ratio": aspect_ratio,
            "k_ext_est": k_ext_est, "surface_area": cum_area[-1] if len(cum_area) > 0 else 0.0}


def get_doy(date_str, year=2018):
    return datetime.datetime(year, int(date_str[:2]), int(date_str[2:])).timetuple().tm_yday


def calculate_sun_vectors(date_str):
    doy = get_doy(date_str)
    delta = 23.45 * math.sin(math.radians(360 / 365 * (doy + 284)))
    lat_rad, delta_rad = math.radians(NEAU_LAT), math.radians(delta)
    vecs = []
    for h in TIME_STEPS:
        omega_rad = math.radians((h - 12) * 15)
        sin_el = math.sin(lat_rad) * math.sin(delta_rad) + math.cos(lat_rad) * math.cos(delta_rad) * math.cos(omega_rad)
        el_rad = math.asin(sin_el)
        if math.degrees(el_rad) <= 0:
            vecs.append(np.array([0, 0, 1]))
            continue
        val = max(-1.0, min(1.0, (math.sin(el_rad) * math.sin(lat_rad) - math.sin(delta_rad)) / (
                    math.cos(el_rad) * math.cos(lat_rad))))
        az_rad = math.acos(val) if h <= 12 else 2 * math.pi - math.acos(val)
        vecs.append(
            np.array([math.cos(el_rad) * math.sin(az_rad), math.cos(el_rad) * math.cos(az_rad), math.sin(el_rad)]))
    return vecs


# ==========================================
# 4. 4D 真实空间演化仿真
# ==========================================
def generate_search_space(variety):
    """
    生成二维虚拟响应面 + 50 cm 固定行距实测切片。

    逻辑：
    1) 前 NUM_CONFIGS 个点仍然是原始二维虚拟搜索空间，用于 RF surrogate 响应面学习；
    2) 额外追加 5 个点：Row_Spacing = 50 cm，Plant_Spacing = 7/8/9/10/11 cm，
       这些点是已有田间实测处理在二维响应面中的 measured transect。

    注意：这里不改变原始 300 个二维随机仿真点，只是在末尾追加 50 cm 实测切片，
    方便后续可视化时把“二维响应面”和“固定行距株距测试”连起来。
    """
    r_virtual = np.concatenate([np.random.uniform(15, 75, int(NUM_CONFIGS * 0.3)),
                                np.random.normal(35, 10, int(NUM_CONFIGS * 0.7)).clip(15, 75)])
    p_virtual = (10000.0 / (r_virtual * TRUTH_DENSITY[variety])) * np.random.normal(1.0, 0.1, NUM_CONFIGS)
    p_virtual = p_virtual.clip(4, 30)

    r_field = np.full(len(MEASURED_PLANT_SPACINGS_CM), MEASURED_ROW_SPACING_CM, dtype=float)
    p_field = MEASURED_PLANT_SPACINGS_CM.astype(float)

    r_arr = np.concatenate([r_virtual, r_field])
    p_arr = np.concatenate([p_virtual, p_field])

    config_type = np.array(["Virtual_2D_response_surface"] * len(r_virtual) +
                           ["Measured_50cm_transect"] * len(r_field), dtype=object)
    true_yield = np.concatenate([np.full(len(r_virtual), np.nan), PAPER_DATA[variety]['y'].astype(float)])
    treatment_index = np.concatenate([np.full(len(r_virtual), np.nan), np.arange(1, len(r_field) + 1, dtype=float)])

    return r_arr, p_arr, config_type, true_yield, treatment_index


def run_4d_simulation():
    variety = FOUR_D_CONFIG['variety']
    dates, flip_ax = FOUR_D_CONFIG['dates'], FOUR_D_CONFIG['flip_axis']
    r_arr, p_arr, config_type, true_yield_arr, treatment_index_arr = generate_search_space(variety)
    total_configs = len(r_arr)

    time_series_data = []
    pheno_records = {}
    min_dist_truth = 10000.0 / (50.0 * TRUTH_DENSITY[variety])

    print(f"\n[{time.strftime('%H:%M:%S')}] 启动 4D 射线追踪 (耦合二维空间距离驱动 SAS 异速生长)...")

    for date_str in dates:
        file_path = os.path.join(BASE_DIR, YEAR_SELECT, f"{variety}_{date_str}.ply")
        if not os.path.exists(file_path):
            file_path = os.path.join(BASE_DIR, f"{variety}_{date_str}.ply")
            if not os.path.exists(file_path): continue

        mesh = load_mesh(file_path, flip_axis=flip_ax)
        pheno_records[date_str] = analyze_plant_phenotype(mesh)
        sun_vecs = calculate_sun_vectors(date_str)

        try:
            intersector_type, use_embree = trimesh.ray.ray_pyembree.RayMeshIntersector, True
        except:
            use_embree = False

        date_lies = []
        for i in tqdm(range(total_configs), desc=f"扫描 {date_str} 时空拓扑", leave=False):
            r, p = r_arr[i], p_arr[i]
            min_dist_current = min(r, p)
            sas_ratio_spatial = min_dist_truth / min_dist_current

            sas_z = np.clip(1.0 + 0.15 * np.log(sas_ratio_spatial + 1e-5), 0.8, 1.4)
            sas_xy = np.clip(1.0 - 0.08 * np.log(sas_ratio_spatial + 1e-5), 0.75, 1.15)

            sub_meshes = []
            for row in range(int(FIELD_SIZE[0] / r) + 1):
                for col in range(int(FIELD_SIZE[1] / p) + 1):
                    xp, yp = -(FIELD_SIZE[0] / r * r) / 2 + row * r, -(FIELD_SIZE[1] / p * p) / 2 + col * p
                    plant = mesh.copy()
                    local_shrink = 0.1 * np.random.uniform(0.95, 1.05)
                    sas_matrix = np.diag([local_shrink * sas_xy, local_shrink * sas_xy, local_shrink * sas_z, 1.0])
                    plant.apply_transform(sas_matrix)
                    random_angle = np.random.uniform(0, 2 * np.pi)
                    plant.apply_transform(trimesh.transformations.rotation_matrix(random_angle, [0, 0, 1]))
                    plant.apply_translation([xp, yp, 0])
                    sub_meshes.append(plant)

            scene = trimesh.util.concatenate(sub_meshes)
            intersector = intersector_type(scene) if use_embree else scene.ray
            daily_hits = 0
            z_top = scene.bounds[1][2] + 10

            for sv in sun_vecs:
                origins = np.column_stack([np.random.uniform(-CORE_ZONE[0] / 2, CORE_ZONE[0] / 2, RAY_COUNT_PER_STEP),
                                           np.random.uniform(-CORE_ZONE[1] / 2, CORE_ZONE[1] / 2, RAY_COUNT_PER_STEP),
                                           np.full(RAY_COUNT_PER_STEP, z_top)])
                _, idx_ray = intersector.intersects_id(origins, np.tile(-sv, (RAY_COUNT_PER_STEP, 1)))
                daily_hits += len(np.unique(idx_ray))

            date_lies.append(daily_hits / (RAY_COUNT_PER_STEP * len(TIME_STEPS)))
        time_series_data.append({'date': date_str, 'doy': get_doy(date_str), 'lies': date_lies})

    print(f"\n[{time.strftime('%H:%M:%S')}] 计算全生育期累积光合积分与动态库限制惩罚...")
    if not pheno_records: raise FileNotFoundError(f"❌ 错误：未能读取到任何 {variety} 的 .ply 模型文件！")

    peak_date = max(pheno_records.keys(), key=lambda d: pheno_records[d]['surface_area'])
    peak_k, peak_ar = pheno_records[peak_date]['k_ext_est'], pheno_records[peak_date]['aspect_ratio']
    base_alpha = 25.0 + 35.0 * peak_k
    alpha = max(25.0, min(75.0, base_alpha * (1.2 / peak_ar)))
    lpp_th = (0.85 + (peak_k - 0.45) * 0.2) / TRUTH_DENSITY[variety]
    doys = [d['doy'] for d in time_series_data]
    total_days = doys[-1] - doys[0]
    a_coef, b_coef = get_empirical_params(variety)

    final_results = []
    for i in range(total_configs):
        r, p = r_arr[i], p_arr[i]
        density = 10000 / (r * p)
        lies_over_time = [d['lies'][i] for d in time_series_data]
        cumulative_lie = np.trapz(lies_over_time, doys)
        peak_lie = lies_over_time[doys.index(get_doy(peak_date))]
        hi = HI_MAX / (1.0 + np.exp(alpha * (lpp_th - (peak_lie / density))))

        min_dist_current = min(r, p)
        sas_ratio_spatial_i = min_dist_truth / min_dist_current
        sas_xy_i = np.clip(1.0 - 0.08 * np.log(sas_ratio_spatial_i + 1e-5), 0.75, 1.15)

        current_dia_cm = pheno_records[peak_date]['raw_width'] * 0.1 * sas_xy_i
        overlap_ratio = max(0, (current_dia_cm - min(r, p)) / current_dia_cm)
        competition_factor = np.exp(-b_coef * density * (overlap_ratio * 2.5))

        yield_score = cumulative_lie * (RUE_CONST / 100.0) * hi * competition_factor
        final_results.append({
            'Config_Type': config_type[i],
            'Treatment_Index': treatment_index_arr[i],
            'Row_Spacing': r, 'Plant_Spacing': p, 'Density': density,
            'True_Yield': true_yield_arr[i],
            'LIE': cumulative_lie / total_days, 'Cum_LIE': cumulative_lie,
            'Peak_LIE': peak_lie,
            'HI': hi,
            'Overlap_Ratio': overlap_ratio,
            'Competition_Factor': competition_factor,
            'Base_Score_No_Competition': cumulative_lie * (RUE_CONST / 100.0) * hi,
            'Yield_Score': yield_score,
            'LIE_Trajectory': lies_over_time
        })

    return pd.DataFrame(final_results), doys, time_series_data, pheno_records, peak_date


if __name__ == "__main__":
    print(f"\n=============================================")
    print(f"  🌱 启动 4D 源库演化引擎 (二维响应面 + 50cm 实测切片) ")
    print(f"=============================================\n")

    variety = FOUR_D_CONFIG['variety']
    df_results, doys, ts_data, pheno_records, peak_date = run_4d_simulation()

    # 同步导出 CSV，便于检查二维响应面与 50 cm 实测切片是否已连通
    df_results.to_csv(f"{variety}_all_simulation_with_50cm_transect.csv", index=False, encoding="utf-8-sig")
    df_results[df_results['Config_Type'] == 'Measured_50cm_transect'].to_csv(
        f"{variety}_measured_50cm_transect_simulation.csv", index=False, encoding="utf-8-sig"
    )

    # 【核心改动】：将昂贵计算的结果打包存起来
    output_filename = f"{variety}_sim_data.pkl"
    with open(output_filename, "wb") as f:
        pickle.dump({
            "df_results": df_results,
            "doys": doys,
            "ts_data": ts_data,
            "pheno_records": pheno_records,
            "peak_date": peak_date
        }, f)

    print(f"\n✅ [{time.strftime('%H:%M:%S')}] 仿真结束！数据已持久化至: {output_filename}")
    print(f"请运行 step2_visualization.py 来调整和生成图表。")