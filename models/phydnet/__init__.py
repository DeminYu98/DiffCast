from .phydnet import PhyDNet_Model


def get_model(in_shape, T_in, T_out, **kwargs):
    return PhyDNet_Model(in_shape, T_in=T_in, T_out=T_out, **kwargs)