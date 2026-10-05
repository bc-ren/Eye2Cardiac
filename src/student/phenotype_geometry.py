"""Source-extracted physical geometry. Public reporting filters to Mesh11.

QC and internal legacy descriptors are retained so numerical semantics do not change.
No patient data loader is bundled. geometry_bridge supplies each mesh explicitly.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy.spatial import ConvexHull, cKDTree, distance
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
WORKER: dict[str, Any] = {}

def signed_volumes_ml(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    output = np.empty(points.shape[0], dtype=np.float64)
    for frame in range(points.shape[0]):
        triangles = np.asarray(points[frame, faces], dtype=np.float64)
        output[frame] = np.sum(
            triangles[:, 0] * np.cross(triangles[:, 1], triangles[:, 2]), axis=None
        ) / 6000.0
    return output


def vertex_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(points[faces], dtype=np.float64)
    area = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    output = np.zeros(len(points), dtype=np.float64)
    share = area / 3.0
    np.add.at(output, faces[:, 0], share)
    np.add.at(output, faces[:, 1], share)
    np.add.at(output, faces[:, 2], share)
    return output


def make_polydata(points: np.ndarray, faces: np.ndarray) -> vtk.vtkPolyData:
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.asarray(points, dtype=np.float64), deep=True))
    packed = np.column_stack(
        (np.full(len(faces), 3, dtype=np.int64), np.asarray(faces, dtype=np.int64))
    ).ravel()
    cells = vtk.vtkCellArray()
    cells.ImportLegacyFormat(numpy_to_vtkIdTypeArray(packed, deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetPolys(cells)
    return poly


def closest_surface_distance(
    query: np.ndarray, surface: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    implicit = vtk.vtkImplicitPolyDataDistance()
    implicit.SetInput(make_polydata(surface, faces))
    return np.fromiter(
        (abs(implicit.EvaluateFunction(point)) for point in np.asarray(query, dtype=np.float64)),
        dtype=np.float64,
        count=len(query),
    )


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(sorted_weights)
    if not len(sorted_values) or cumulative[-1] <= 0:
        return float("nan")
    return float(np.interp(q * cumulative[-1], cumulative, sorted_values))


def load_topology(config: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(config["teacher_topology"], allow_pickle=False) as values:
        output = {
            "lv_ids": np.asarray(values["LV_global_ids"], dtype=np.int64),
            "myo_ids": np.asarray(values["Myo_global_ids"], dtype=np.int64),
            "lv_faces": np.asarray(values["LV_faces"], dtype=np.int64),
            "myo_faces": np.asarray(values["Myo_faces"], dtype=np.int64),
        }
    with np.load(config["full_topology"], allow_pickle=False) as values:
        labels = np.asarray(values["full_labels"], dtype=np.int64)
        template = np.asarray(values["template_points"], dtype=np.float64)
    la_ids = np.flatnonzero(labels == 4)
    template_lv = template[output["lv_ids"]]
    template_la = template[la_ids]
    distances, _ = cKDTree(template_la).query(template_lv, k=1, workers=1)
    threshold = float(config["phenotype"]["template_mitral_contact_threshold_mm"])
    output["basal_ring_ids"] = np.flatnonzero(distances <= threshold).astype(np.int64)
    if not (50 <= len(output["basal_ring_ids"]) <= 1500):
        raise ValueError("Fixed template basal-ring vertex count is implausible")
    if len(output["lv_ids"]) != 6141 or len(output["myo_ids"]) != 6141:
        raise ValueError("Unexpected LV/Myo topology")
    if not np.array_equal(output["lv_faces"], output["myo_faces"]):
        raise ValueError("LV and Myo surface topology differs")
    return output


def derive_axis(lv: np.ndarray, basal_ring_ids: np.ndarray) -> dict[str, Any]:
    ring = np.asarray(lv[basal_ring_ids], dtype=np.float64)
    center = ring.mean(axis=0)
    centered = ring - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if float(np.dot(lv.mean(axis=0) - center, normal)) < 0:
        normal = -normal
    longitudinal = (lv - center) @ normal
    apex_index = int(np.argmax(longitudinal))
    length = float(longitudinal[apex_index])
    residual_p95 = float(np.quantile(np.abs(centered @ normal), 0.95))
    return {
        "center": center,
        "normal": normal,
        "longitudinal": longitudinal,
        "apex_index": apex_index,
        "length_mm": length,
        "plane_residual_p95_mm": residual_p95,
    }


def maximum_short_axis_diameter(lv: np.ndarray, axis: dict[str, Any]) -> float:
    normal = axis["normal"]
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(helper, normal))) > 0.85:
        helper = np.array([0.0, 1.0, 0.0])
    u = helper - float(np.dot(helper, normal)) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    longitudinal = axis["longitudinal"]
    length = float(axis["length_mm"])
    xy = np.column_stack(((lv - axis["center"]) @ u, (lv - axis["center"]) @ v))
    best = float("nan")
    half_width = max(1.5, 0.025 * length)
    for level in np.linspace(0.10 * length, 0.90 * length, 33):
        selected = np.abs(longitudinal - level) <= half_width
        points = xy[selected]
        if len(points) < 12:
            continue
        try:
            hull_points = points[ConvexHull(points).vertices]
            value = float(distance.pdist(hull_points).max()) if len(hull_points) > 1 else float("nan")
        except Exception:
            continue
        if np.isfinite(value) and (not np.isfinite(best) or value > best):
            best = value
    return best


def frame_geometry(
    lv: np.ndarray,
    myo: np.ndarray,
    lv_faces: np.ndarray,
    myo_faces: np.ndarray,
    basal_ring_ids: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    settings = config["phenotype"]
    axis = derive_axis(lv, basal_ring_ids)
    thickness = closest_surface_distance(lv, myo, myo_faces)
    areas = vertex_areas(lv, lv_faces)
    length = axis["length_mm"]
    basal = np.zeros(len(lv), dtype=bool)
    basal[basal_ring_ids] = True
    valid = (
        (axis["longitudinal"] > float(settings["basal_exclusion_fraction"]) * length)
        & (axis["longitudinal"] < (1.0 - float(settings["apical_exclusion_fraction"])) * length)
        & (~basal)
        & (thickness >= float(settings["wall_thickness_min_mm"]))
        & (thickness <= float(settings["wall_thickness_max_mm"]))
        & np.isfinite(thickness)
        & np.isfinite(areas)
        & (areas > 0)
    )
    valid_area_fraction = float(areas[valid].sum() / areas.sum()) if areas.sum() > 0 else 0.0
    mean_thickness = float(np.average(thickness[valid], weights=areas[valid])) if valid.any() else float("nan")
    p95_thickness = weighted_quantile(thickness[valid], areas[valid], 0.95) if valid.any() else float("nan")
    short_axis = maximum_short_axis_diameter(lv, axis)
    sphericity = short_axis / length if np.isfinite(short_axis) and length > 0 else float("nan")
    return {
        "axis": axis,
        "thickness": thickness,
        "areas": areas,
        "valid": valid,
        "valid_area_fraction": valid_area_fraction,
        "mean_thickness_mm": mean_thickness,
        "p95_thickness_mm": p95_thickness,
        "short_axis_mm": short_axis,
        "sphericity": sphericity,
    }


def worker_init(
    config: dict[str, Any], topology: dict[str, np.ndarray], teacher_lookup: dict[str, dict[str, Any]]
) -> None:
    WORKER.clear()
    WORKER.update(config=config, topology=topology, teacher_lookup=teacher_lookup)


def process_record(row: tuple[int, str, str, str]) -> dict[str, Any]:
    row_id, cohort, split, eid = row
    config = WORKER["config"]
    top = WORKER["topology"]
    teacher_row = WORKER["teacher_lookup"].get(eid)
    result: dict[str, Any] = {
        "row_id": row_id,
        "cohort": cohort,
        "split": split,
        "eid": eid,
        "coordinate_system": config["coordinate_system"],
        "mesh_source_role": config["mesh_role"],
        "teacher_release_member": teacher_row is not None,
        "teacher_source_split": None if teacher_row is None else teacher_row["split"],
        "phenotype_qc": "FAIL",
        "phenotype_failure_reason": "",
    }
    try:
        mesh, source_path, original_ed = load_teacher_target_mesh(eid, teacher_row, top, config)
        result["mesh_source_path"] = source_path
        result["original_ed_index"] = original_ed
        if not np.isfinite(mesh).all():
            raise ValueError("nonfinite_teacher_target_mesh")
        lv, myo = np.asarray(mesh[:, 0], dtype=np.float64), np.asarray(mesh[:, 1], dtype=np.float64)
        lv_signed = signed_volumes_ml(lv, top["lv_faces"])
        myo_signed = signed_volumes_ml(myo, top["myo_faces"])
        if (lv_signed <= 0).any() or (myo_signed <= 0).any():
            raise ValueError("nonpositive_signed_surface_volume")
        lv_volume = np.abs(lv_signed)
        myo_enclosed_volume = np.abs(myo_signed)
        tissue_volume = myo_enclosed_volume - lv_volume
        if (tissue_volume <= 0).any():
            raise ValueError("myocardial_enclosed_volume_not_larger_than_lv")
        ed = 0
        actual_ed = int(np.argmax(lv_volume))
        es = int(np.argmin(lv_volume))
        if actual_ed != 0:
            raise ValueError(f"teacher_preprocessing_ed_not_zero actual={actual_ed}")
        ed_geometry = frame_geometry(lv[ed], myo[ed], top["lv_faces"], top["myo_faces"], top["basal_ring_ids"], config)
        es_geometry = frame_geometry(lv[es], myo[es], top["lv_faces"], top["myo_faces"], top["basal_ring_ids"], config)
        common = ed_geometry["valid"] & es_geometry["valid"]
        common_area_fraction = float(ed_geometry["areas"][common].sum() / ed_geometry["areas"].sum()) if common.any() else 0.0
        global_wall_thickening = float("nan")
        if common.any():
            local = 100.0 * (
                es_geometry["thickness"][common] - ed_geometry["thickness"][common]
            ) / ed_geometry["thickness"][common]
            global_wall_thickening = float(
                np.average(local, weights=ed_geometry["areas"][common])
            )
        ed_axis = ed_geometry["axis"]
        es_axis = es_geometry["axis"]
        cosine = abs(float(np.dot(ed_axis["normal"], es_axis["normal"])))
        axis_angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        longitudinal_shortening = 100.0 * (
            ed_axis["length_mm"] - es_axis["length_mm"]
        ) / ed_axis["length_mm"]
        edv = float(lv_volume[ed])
        esv = float(lv_volume[es])
        result.update({
            "mesh_ed_frame": ed,
            "mesh_es_frame": es,
            "mesh_es_original_index": int((original_ed + es) % 50),
            "lv_edv_ml": edv,
            "lv_esv_ml": esv,
            "lv_sv_ml": edv - esv,
            "lv_ef_pct": 100.0 * (edv - esv) / edv,
            "lv_mass_g": float(tissue_volume[ed]) * float(config["phenotype"]["myocardial_density_g_per_ml"]),
            "lv_mean_wall_thickness_ed_mm": ed_geometry["mean_thickness_mm"],
            "lv_mean_wall_thickness_es_mm": es_geometry["mean_thickness_mm"],
            "lv_wall_thickness_p95_ed_mm": ed_geometry["p95_thickness_mm"],
            "lv_sphericity_index_ed_ratio": ed_geometry["sphericity"],
            "lv_sphericity_index_es_ratio": es_geometry["sphericity"],
            "lv_global_wall_thickening_pct": global_wall_thickening,
            "lv_base_apex_length_ed_mm": ed_axis["length_mm"],
            "lv_base_apex_length_es_mm": es_axis["length_mm"],
            "lv_longitudinal_shortening_pct": longitudinal_shortening,
            "lv_short_axis_diameter_ed_mm": ed_geometry["short_axis_mm"],
            "lv_short_axis_diameter_es_mm": es_geometry["short_axis_mm"],
            "lv_wall_valid_area_fraction_ed": ed_geometry["valid_area_fraction"],
            "lv_wall_valid_area_fraction_es": es_geometry["valid_area_fraction"],
            "lv_wall_common_area_fraction_ed_es": common_area_fraction,
            "lv_ed_es_long_axis_angle_deg": axis_angle,
            "lv_basal_plane_residual_p95_ed_mm": ed_axis["plane_residual_p95_mm"],
            "lv_basal_plane_residual_p95_es_mm": es_axis["plane_residual_p95_mm"],
        })
        settings = config["phenotype"]
        reasons: list[str] = []
        if not (float(settings["wall_valid_area_fraction_min"]) <= ed_geometry["valid_area_fraction"]):
            reasons.append("ed_wall_valid_area_fraction_low")
        if not (float(settings["wall_valid_area_fraction_min"]) <= es_geometry["valid_area_fraction"]):
            reasons.append("es_wall_valid_area_fraction_low")
        if common_area_fraction < float(settings["wall_valid_area_fraction_min"]):
            reasons.append("ed_es_common_wall_area_fraction_low")
        for phase, geometry in (("ed", ed_geometry), ("es", es_geometry)):
            if not (float(settings["long_axis_min_mm"]) <= geometry["axis"]["length_mm"] <= float(settings["long_axis_max_mm"])):
                reasons.append(f"{phase}_base_apex_length_outside_range")
            if not (float(settings["short_axis_min_mm"]) <= geometry["short_axis_mm"] <= float(settings["short_axis_max_mm"])):
                reasons.append(f"{phase}_short_axis_outside_range")
            if not (float(settings["sphericity_min"]) <= geometry["sphericity"] <= float(settings["sphericity_max"])):
                reasons.append(f"{phase}_sphericity_outside_range")
        if axis_angle > float(settings["ed_es_axis_angle_max_deg"]):
            reasons.append("ed_es_long_axis_angle_high")
        if not (float(settings["global_wall_thickening_min_pct"]) <= global_wall_thickening <= float(settings["global_wall_thickening_max_pct"])):
            reasons.append("global_wall_thickening_outside_range")
        result["phenotype_qc"] = "PASS" if not reasons else "FAIL"
        result["phenotype_failure_reason"] = ";".join(reasons)
    except Exception as exc:
        result["phenotype_failure_reason"] = f"{type(exc).__name__}:{exc}"
    return result

