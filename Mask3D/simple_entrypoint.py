from models_without_hydra.mask3d import Mask3D


model = Mask3D(
    hidden_dim=128,
    num_queries=100,
    num_heads=8,
    dim_feedforward=1024,
    sample_sizes=[200, 800, 3200, 12800, 51200],
    shared_decoder=True,
    num_classes=19,
    num_decoders=3,
    dropout=0.0,
    pre_norm=False,
    positional_encoding_type="fourier",
    non_parametric_queries=True,
    train_on_segments=False,
    normalize_pos_enc=True,
    use_level_embed=False,
    scatter_type="mean",
    hlevels=[0, 1, 2, 3],
    use_np_features=False,
    voxel_size=0.02,
    max_sample_size=False,
    random_queries=False,
    gauss_scale=1.0,
    random_query_both=False,
    random_normal=False,
    backbone_in_channels=3,
    backbone_out_channels=20,
    backbone_dilations=[1, 1, 1, 1],
    backbone_bn_momentum=0.02,
    backbone_conv1_kernel_size=5
)
print("Created model!")
