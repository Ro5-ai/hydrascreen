import torch


class ActivitySmoother:
    def __init__(self, gsm: float = 1 / 4, std_max: float = 5 / 2):
        self.gsm = gsm
        self.std_max = std_max

    @staticmethod
    def outlier_removal(x: torch.Tensor, std_max: float) -> torch.Tensor:
        # Localize
        is_active = x != 0
        z = x[is_active]

        # Reduce tail points
        is_too_small = z < -std_max
        z[is_too_small] = -std_max

        is_too_large = z > std_max
        z[is_too_large] = std_max

        # Re-assign to return in x space
        x[is_active] = z
        return x

    @staticmethod
    def gaussian_smoothing(x: torch.Tensor, gsm: float) -> torch.Tensor:
        is_active = x != 0
        noise = torch.randn(sum(is_active), device=x.device) * gsm
        x[is_active] += noise
        return x

    def __call__(self, activity: torch.Tensor) -> torch.Tensor:
        """Smooth activity values according to Gaussian smoothing and outlier removal

        Args:
            activity (torch.Tensor): Shape [N]. Activity values to be smoothed

        Returns:
            torch.Tensor: Shape [N]. Smoothed activity values
        """
        x = self.outlier_removal(activity, self.std_max)
        x = self.gaussian_smoothing(x, self.gsm)
        return x


class Preprocessor(torch.nn.Module):
    def __init__(
        self, norm_config: dict, mirror_prob=0.5, gsm: float = 1 / 4, std_max: float = 5 / 2
    ):
        """Preprocessor to manage data labelling during training
        Args:
            norm_config (dict): Configuration of the statistics
            mirror_prob (float, optional): Probability of mirroring the voxels. Defaults to 0.5.
            gsm (float, optional): Gaussian smoothing factor, more or less this should be ~ half the resolution of the IC50 reading. Defaults to 1/4.
            std_max (float, optional): Max std to observe in the data. Defaults to 3.
        """
        super(Preprocessor, self).__init__()
        self.norm_config = norm_config
        self.mirror_prob = mirror_prob
        self.mirror_dist = torch.distributions.Bernoulli(mirror_prob)
        # Use effective GSM because smoother operates in Z space
        effective_gsm = gsm / self.norm_config["affinity"]["std"]
        self.activity_smoother = ActivitySmoother(gsm=effective_gsm, std_max=std_max)

    def normalise(self, activity: torch.Tensor, rmsd: torch.Tensor, dock: torch.Tensor):
        """Post processor for structure-based predictions - training.

        Args:
            activity (torch.Tensor): activity tensor (float) shape Batch Size
            rmsd (torch.Tensor): rmsd tensor (float) shape Batch Size
            dock (torch.Tensor): dock energy tensor (float) shape Batch Size
        Returns:
            [tuple]: activity, rmsd, dock
        """
        # Normalise Activity. Only for non-zero values.
        activity = torch.abs(activity)
        is_active = activity > 0
        activity = (activity - self.norm_config["affinity"]["mean"]) / self.norm_config["affinity"][
            "std"
        ]

        # Smoothing
        if self.activity_smoother is not None:
            # NOTE that the smoother is operating in standard space! It is normalized
            activity = self.activity_smoother(activity)
        activity[~is_active] = 0

        # Normalise RMSD & Zero out rmsd values that are above the threshold between [-1 and 1]
        rmsd = (rmsd - self.norm_config["rmsd"]["mean"]) / self.norm_config["rmsd"]["std"]
        rmsd[rmsd > 1] = 1
        rmsd[rmsd < -1] = -1

        # Normalise RMSD & Zero out dock scores above 0.
        is_dock = dock < 0
        dock = (dock - self.norm_config["docking"]["mean"]) / self.norm_config["docking"]["std"]
        dock[~is_dock] = 0
        return activity, rmsd, dock

    def denormalise_preds(self, activity: torch.Tensor, rmsd: torch.Tensor, dock: torch.Tensor):
        """Denormalisatiobn Post processor for structure-based predictions - inference

        Args:
            activity (torch.Tensor): activity tensor (float) shape Batch Size
            rmsd (torch.Tensor): rmsd tensor (float) shape Batch Size
            dock (torch.Tensor): dock energy tensor (float) shape Batch Size
        Returns:
            [tuple]: activty, rmsd, dock
        """
        activity = (
            activity * self.norm_config["affinity"]["std"] + self.norm_config["affinity"]["mean"]
        )
        rmsd = rmsd * self.norm_config["rmsd"]["std"] + self.norm_config["rmsd"]["mean"]
        dock = dock * self.norm_config["docking"]["std"] + self.norm_config["docking"]["mean"]
        return activity, rmsd, dock

    def mirror_transform_voxels(self, voxels: torch.Tensor):
        """Transform based on mirror translations in the voxels accross the last 3 channels ONLY.

        Args:
            voxels (torch.Tensor): Voxels shape [BATCH_SIZE, NUM_CHANNELS, NX, NY, NZ]

        Returns:
            _type_: _description_
        """
        shape = voxels.shape
        assert shape[-1] == shape[-2] == shape[-3], "Can only mirror isomorphic voxels"
        surpluss = len(shape) - 3
        rotate_axes = self.mirror_dist.sample([3])
        flip_dims = [x.item() + surpluss for x in torch.where(rotate_axes == 1)[0]]
        return voxels.flip(flip_dims)

    def forward(self, voxels, activity, rmsd, dock):
        activity, rmsd, dock = self.normalise(activity=activity, rmsd=rmsd, dock=dock)
        voxels = self.mirror_transform_voxels(voxels=voxels)
        return voxels, activity, rmsd, dock
