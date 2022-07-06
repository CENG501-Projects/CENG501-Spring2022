import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from ShadowRemoverNetwork import ShadowRemoverNetwork
from dataset import dataset
import matplotlib.image as img
from errorCalculator import peak_signal_noise_ratio, mean_squared_error, structural_sim

transformation = torchvision.transforms.Compose((torchvision.transforms.Resize([128,128]),torchvision.transforms.ToTensor()))
test_dataset = dataset(image_dir="ISTD_Dataset/test/test_A", mask_dir="ISTD_Dataset/test/test_B", groud_truth_dir="ISTD_Dataset/test/test_C",transformation=transformation)
test_loader = DataLoader(test_dataset,shuffle=True,num_workers=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_results_path = 'Results_ISTD/'

enlarge = torchvision.transforms.Resize([256,256])

if __name__ == '__main__':
    model = ShadowRemoverNetwork().to(device)
    print('#generator parameters:',sum(param.numel() for param in model.parameters()))
    model.load_state_dict(torch.load("ShadowRemoverNetwork_128.ckpt", map_location=torch.device(device)))
    model.eval()
    total_mse = 0
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            I_s, M, gt = data
            I_ns_history, _ = model(I_s, M)
            out = torch.clamp(I_ns_history[-1], 0., 1.)
            total_mse += mean_squared_error(out, gt)
            total_psnr += peak_signal_noise_ratio(out, gt)
            total_ssim += structural_sim(out, gt)
            out = np.squeeze(enlarge(I_ns_history[-1]).cpu().numpy())
            out = out.transpose((1,2,0))
            ground_truth = np.squeeze(enlarge(gt).cpu().numpy())
            ground_truth = ground_truth.transpose((1,2,0))
            shadow_image = np.squeeze(enlarge(I_s).cpu().numpy())
            shadow_image = shadow_image.transpose((1,2,0))
            img.imsave(test_results_path+str(i)+"_result.jpeg",np.concatenate((np.uint8(shadow_image*255.),np.uint8(out*255.),np.uint8(ground_truth*255)),1))
            
    ssim = total_ssim/len(test_loader)
    rmse = (total_mse/len(test_loader))**0.5
    psnr = total_psnr/len(test_loader)
    print('Root mean squared error on ISTD dataset',rmse)
    print('Average peak signal noise ratio on ISTD dataset',psnr)
    print('Average structural similarity on ISTD dataset',ssim)