"""Framingham general CHD risk, extracted unchanged from the experiment source.

Uses mg/dL lipids and mmHg blood pressure; callers convert mmol/L to mg/dL.
The published 10-year score is a rank comparator in the diagnosis task.
"""
import numpy as np


def framingham_chd(age, female, total_chol, hdl, sbp, dbp, diabetes, smoker):
    """Wilson 1998 Framingham general CHD score (categorical lipids/BP)."""
    tc_cat = np.select(
        [total_chol < 160, total_chol < 200, total_chol < 240,
         total_chol < 280], [0, 1, 2, 3], default=4)
    hdl_cat = np.select(
        [hdl < 35, hdl < 45, hdl < 50, hdl < 60], [0, 1, 2, 3], default=4)
    # JNC-V category is the worse of systolic and diastolic categories.
    sbp_cat = np.select([sbp < 120, sbp < 130, sbp < 140, sbp < 160],
                        [0, 1, 2, 3], default=4)
    dbp_cat = np.select([dbp < 80, dbp < 85, dbp < 90, dbp < 100],
                        [0, 1, 2, 3], default=4)
    bp_cat = np.maximum(sbp_cat, dbp_cat)
    male_tc = np.array([-0.65945, 0.0, 0.17692, 0.50539, 0.65713])
    female_tc = np.array([-0.26138, 0.0, 0.20771, 0.24385, 0.53513])
    male_hdl = np.array([0.49744, 0.24310, 0.0, -0.05107, -0.48660])
    female_hdl = np.array([0.84312, 0.37796, 0.19785, 0.0, -0.42951])
    male_bp = np.array([-0.00226, 0.0, 0.28320, 0.52168, 0.61859])
    female_bp = np.array([-0.53363, 0.0, -0.06773, 0.26288, 0.46573])
    lp_m = (0.04826 * age + male_tc[tc_cat] + male_hdl[hdl_cat]
            + male_bp[bp_cat] + 0.42839 * diabetes + 0.52337 * smoker)
    lp_f = (0.33766 * age - 0.00268 * age * age + female_tc[tc_cat]
            + female_hdl[hdl_cat] + female_bp[bp_cat]
            + 0.59626 * diabetes + 0.29246 * smoker)
    risk_m = 1.0 - 0.90015 ** np.exp(lp_m - 3.09750)
    risk_f = 1.0 - 0.96246 ** np.exp(lp_f - 9.92545)
    return np.where(female, risk_f, risk_m)
