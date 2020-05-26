import torch

from fairmot.modeling.networks.pose_dla_dcn import get_pose_net


def create_model():
    heads = {'hm': 1, 'wh': 2, 'id': 512, 'reg': 2}
    head_conv = 256
    model = get_pose_net(num_layers=34, heads=heads, head_conv=head_conv)
    return model