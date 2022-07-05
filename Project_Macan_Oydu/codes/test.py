import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from ShadowRemoverNetwork import ShadowRemoverNetwork
from dataset import dataset
import matplotlib.image as img
from errorCalculator import peak_signal_noise_ratio, mean_squared_error

transformation = torchvision.transforms.Compose((torchvision.transforms.Resize([128,128]),torchvision.transforms.ToTensor()))
test_dataset = dataset(image_dir="ISTD_Dataset/test/test_A", mask_dir="ISTD_Dataset/test/test_B", groud_truth_dir="ISTD_Dataset/test/test_C",transformation=transformation)
test_loader = DataLoader(test_dataset,shuffle=True,num_workers=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_results_path = 'Results_ISTD/'

enlarge = torchvision.transforms.Resize([256,256])

if __name__ == '__main__':
    model = ShadowRemoverNetwork().to(device)
    print('#generator parameters:',sum(param.numel() for param in model.parameters()))
    model.load_state_dict(torch.load("ShadowRemoverNetwork.ckpt", map_location=torch.device(device)))
    model.eval()
    total_mse = 0
    total_psnr = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            I_s, M, gt = data
            
            I_ns_history, _ = model(I_s, M)
            out = torch.clamp(I_ns_history[-1], 0., 1.)
            total_mse += mean_squared_error(out, gt)
            total_psnr += peak_signal_noise_ratio(out, gt)
            out = np.squeeze(enlarge(I_ns_history[-1]).cpu().numpy())
            out = out.transpose((1,2,0))
            original_image = np.squeeze(enlarge(gt).cpu().numpy())
            original_image = original_image.transpose((1,2,0))
            
            img.imsave(test_results_path+str(i)+"_result.jpeg",np.concatenate((np.uint8(out*255.),np.uint8(original_image*255)),1))
    rmse = (total_mse/len(test_loader))**0.5
    psnr = total_psnr/len(test_loader)
    print('Root mean squared error on ISTD dataset',rmse)
    print('Average peak signal noise ratio on ISTD dataset',psnr)