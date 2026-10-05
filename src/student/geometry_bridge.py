"""Mesh-only bridge to the original physical phenotype implementation."""
import json
from pathlib import Path
import numpy as np
import phenotype_geometry as PHMOD

PAIR = None


def worker_init():
    config = json.loads(Path(__file__).with_name('phenotypes.json').read_text())
    top = PHMOD.load_topology(config)
    np.testing.assert_array_equal(top['basal_ring_ids'], np.load(config['basal_ring_ids'], allow_pickle=False))
    PHMOD.worker_init(config, top, {})
    PHMOD.load_teacher_target_mesh = lambda eid, row, topology, cfg: (PAIR, 'explicit-mesh-input', 0)


def phenotype_job(payload):
    global PAIR
    index, identifier, PAIR, ed, es = payload
    result = PHMOD.process_record((index, 'UKB-Dev', 'test', identifier))
    result.update(mesh_ed_frame=ed, mesh_es_frame=es, source_role='CFP-time reconstruction')
    return result
