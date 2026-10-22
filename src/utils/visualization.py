from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.utils import save_image
from utils.boundary import calc_edge


def draw_boundary(x, img, color):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = (img-img.min())/(img.max()-img.min())

    edged = calc_edge(x[0,:], mode='canny')
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for idx, imc in enumerate(range(3)):
        img[idx,:,:] = cv2.drawContours(np.uint8(img[idx,:,:]*255), contours, -1, color[idx], 1)/255.
    return img



def save_sampling_results_as_imgs(
    batch_imgs, batch_msks, batch_ids, batch_prds, 
    ensemble_list_of_samples_in_timesteps=None, middle_steps_of_sampling=4,
    save_dir="./saved_imgs",
    dataset_name=None,
    result_id=None,
    img_ext="png",
    save_mat=False
):
    save_directory =  "/".join([d for d in [save_dir, dataset_name, result_id] if d])
    
    # check dir
    Path(save_directory).mkdir(exist_ok=True, parents=True)

    for im, gt, id, pd in zip(batch_imgs, batch_msks, batch_ids, batch_prds):
        sd = f"{save_directory}/{id}"
        Path(sd).mkdir(exist_ok=True)
        
        save_image(im   , f"{sd}/im.{img_ext}")
        save_image(gt[0], f"{sd}/gt.{img_ext}")
        save_image(pd[0], f"{sd}/pd.{img_ext}")
        
        # draw gt-pd on the image
        db = draw_boundary(torch.where(gt>0, 1, 0), im, (0, 255, 0))
        db = draw_boundary(torch.where(pd>0, 1, 0), db, (0, 0, 255))
        save_image(torch.tensor(db), f"{sd}/db.{img_ext}")

        if save_mat:
            torch.save(gt[0], f"{sd}/gt.pt")
            torch.save(pd[0], f"{sd}/pd.pt")
            
    if ensemble_list_of_samples_in_timesteps:
        T = len(ensemble_list_of_samples_in_timesteps[0])
        sample_ts = np.int32(np.round(np.linspace(0, T-1, middle_steps_of_sampling+2))) # added 2 because inxed 0 and T-1
        for i, ensemble_i in enumerate(ensemble_list_of_samples_in_timesteps):
            batch_samples = [ensemble_i[i] for i in sample_ts]
            for batch_sample, t in zip(batch_samples, sample_ts):
                for sample, id in zip(batch_sample, batch_ids):
                    save_image(sample[0], f"{save_directory}/{id}/t_{t}.{img_ext}")

