"""Raw CFP pixels + immutable anatomical products, never cached visual tokens."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import functional as TF
from shared import ROOT, PREP, S, status

ANATOMY_KEYS = ['disc', 'paths', 'segment_mask', 'segment_geometry', 'adjacency']


def pixels(row):
    path = row['preprocessed_rgb'] if row['spatial_complete'] else row['image_path']
    with Image.open(path) as im:
        im = im.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
        if row['side'] == 'L':
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return TF.normalize(TF.to_tensor(im), [.485, .456, .406], [.229, .224, .225]).numpy()


class RawCache:
    """Cache API compatibility: tokens key contains pixels, not retinal features."""
    def __init__(self, scopes=('paired', 'clinical'), smoke=False):
        assert not smoke, 'Raw-image smoke is selected by patient, not legacy token cache'
        self.arrays = []
        self.lookup = {}
        for scope in scopes:
            for shard in range(2):
                p = Path(S['anatomy_cache_root']) / f'{scope}_shard{shard}'
                assert status(p) == 'COMPLETED'
                a = {k: np.load(p / (k + '.npy'), mmap_mode='r') for k in ANATOMY_KEYS}
                ids = np.load(p / 'token_id.npy')
                idx = len(self.arrays)
                self.arrays.append(a)
                for j, token in enumerate(ids):
                    assert int(token) not in self.lookup
                    self.lookup[int(token)] = (idx, j)
        self.manifest = pd.read_parquet(PREP / 'image_manifest_SERVER_ONLY.parquet')
        assert self.manifest.image_path.is_unique and self.manifest.token_id.is_unique
        self.rows = self.manifest.set_index('image_path').to_dict('index')
        self.path_id = dict(zip(self.manifest.image_path, self.manifest.token_id))
        self.blank = {k: np.zeros_like(a[0]) for k, a in self.arrays[0].items()}
        self.blank.update(tokens=np.zeros((3, 224, 224), np.float32), cls=np.zeros(1024, np.float32))

    def get(self, path):
        if pd.isna(path):
            return {k: v.copy() for k, v in self.blank.items()}, False
        path = str(path)
        row = dict(self.rows[path], image_path=path)
        idx, j = self.lookup[int(row['token_id'])]
        result = {k: np.array(v[j], copy=True) for k, v in self.arrays[idx].items()}
        result.update(tokens=pixels(row), cls=np.zeros(1024, np.float32))
        assert np.isfinite(result['tokens']).all()
        return result, True
