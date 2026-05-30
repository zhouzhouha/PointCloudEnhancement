import torch.utils.data as data
from utils.general import *
import trimesh
from trimesh.sample import sample_surface
from scipy.spatial import cKDTree
from tqdm import tqdm
import utils.general as utils


class ReconDataSet(data.Dataset):

    def __init__(self,split,dataset_path,dist_file_name):

        model = trimesh.load(dataset_path)
        if not type(model) == trimesh.PointCloud:
            model = utils.as_mesh(trimesh.load(dataset_path))
            if (model.faces.shape[0] > 0):
                self.points = sample_surface(model,250000)[0]
            else:
                self.points = model.vertices
        else:
            self.points = model.vertices

        self.points = self.points - self.points.mean(0,keepdims=True)
        scale = np.abs(self.points).max()
        self.points = self.points / scale

        sigmas = []

        ptree = cKDTree(self.points)

        for p in tqdm(np.array_split(self.points, 10, axis=0)):
            d = ptree.query(p, int(51.0))
            sigmas.append(d[0][:, -1])

        sigmas = np.concatenate(sigmas)
        
        # Ensure we have at least 250000 samples
        target_samples = 250000
        if sigmas.shape[0] < target_samples:
            repeat_count = target_samples // sigmas.shape[0] + 1
            sigmas = np.tile(sigmas, repeat_count)[:target_samples]
            sigmas_big = 1.0 * np.ones(target_samples)
            pnts = np.tile(self.points, [repeat_count, 1])[:target_samples]
        else:
            # Sample down if we have too many points
            idx = np.random.choice(sigmas.shape[0], target_samples, replace=False)
            sigmas = sigmas[idx]
            sigmas_big = 1.0 * np.ones_like(sigmas)
            pnts = self.points[idx]

        sample = np.concatenate([pnts + np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=pnts.shape),
                                 pnts + np.expand_dims(sigmas_big, -1) * np.random.normal(0.0, 1.0, size=pnts.shape)],
                                axis=0)

        dists = ptree.query(sample)[0]
        self.dists = np.concatenate([sample, np.expand_dims(dists, axis=-1)], axis=-1)
        self.npyfiles_mnfld = [dataset_path]

    def __getitem__(self, index):

        point_set_mnlfld = torch.from_numpy(self.points).float()
        sample_non_mnfld = torch.from_numpy(self.dists).float()
        random_idx = (torch.rand(128**2) * point_set_mnlfld.shape[0]).long()
        point_set_mnlfld = torch.index_select(point_set_mnlfld,0,random_idx)

        random_idx = (torch.rand(128**2) * sample_non_mnfld.shape[0]).long()
        sample_non_mnfld = torch.index_select(sample_non_mnfld,0,random_idx)

        return point_set_mnlfld,sample_non_mnfld,index


    def __len__(self):
        return 1

