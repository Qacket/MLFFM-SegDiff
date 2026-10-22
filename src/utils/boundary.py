import cv2
import numpy as np
import torch


def calc_edge(x, mode='canny'):
    x = np.uint8(x)
    edge = cv2.Canny(image=x, threshold1=0, threshold2=0)
    return edge



def calc_distance_map(x, mode='l2'):
    # if isinstance(x, torch.Tensor):
    #     x = x.numpy()
    
    # Convert the data to grayscale [0,255]
    binary_x = 1 - np.uint8((x-x.min())/(x.max()-x.min()))

    if mode.lower() == 'l1':
        dt_mode = cv2.DIST_L1
    elif mode.lower() == 'l2':
        dt_mode = cv2.DIST_L2
    else:
        raise ValueError("<mode> must be 'l1' or 'l2'.")
    
    # Calculate the distance transform
    dist_transform= cv2.distanceTransform(binary_x, dt_mode, 0)

    return dist_transform



def calc_boundary_att(batch_x, batch_t, T, gamma=1.5, *args, **kwargs):
    """
    Parameters:
        - batch_x : [tensor] |-> input data matrix
        - batch_t : [tensor] |-> current timestep
        - T       : [int]    |-> maximum timesteps
        - gamma   : [float]  |-> sharpness [default is 1.5]
    Output:
        - boundary_att
    """
    
    
    # boundary thickness (max value thickness)
    bt = np.round(batch_x.shape[-1]*0.01)
    
    X = batch_x.detach().cpu().numpy()
    X = (X-X.min())/(X.max()-X.min()) # normalize because X is in range ~ (-1, 1)
    device = batch_x.get_device()
    
    atts = []
    for x, t in zip(X, batch_t):
        if x.sum().item() > X.shape[-1]**2/100.: # foreground area is bigger than 1% of the image
            x = (x-x.min())/(x.max()-x.min())
            edge = calc_edge(x[0,:])
            dist_x = calc_distance_map(edge, mode='l2')
            tmp = X.shape[-1]*1.1415 - dist_x
            normalized_inv_dist_x = (tmp-tmp.min())/ (tmp.max()-tmp.min())
            
            t_p = ((gamma*(T-t.item()))/T)**gamma
            att = normalized_inv_dist_x**t_p            
        else:
            att = np.ones_like(x[0,:])
        atts.append(att)

    atts = np.array(atts)
    
    W = torch.stack([torch.from_numpy(att) for att in atts], dim=0)
    if device != -1:
        W = W.to(device)

    W = torch.unsqueeze(W, dim=1)
    return W

