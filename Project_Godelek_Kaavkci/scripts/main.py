import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio
import numpy as np
import matplotlib.pyplot as plt

import dataset
import networks
import utils
from torchinfo import summary


def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    device = 'cpu'
    hr_images, lr_images, mean_image = dataset.get_images(4)
    model = networks.VDSR_new(mean_image.to(device))
    model.apply(utils.init_weights)
    model.to(device)
    psnr = PeakSignalNoiseRatio()
    hr_images = torch.autograd.Variable(hr_images.float())
    lr_images = torch.autograd.Variable(lr_images.float())
    train_dataset = torch.utils.data.TensorDataset(lr_images, hr_images)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=16,
                                              shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.0001,
                                 betas=(0.99, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500, 2000, 5000, 7000, 9000], gamma=0.5)
    epochs = 1000
    loss_history = []
    for epoch in range(1, epochs):

        temp_loss = []
        for idx, data in enumerate(trainloader, 0):
             inputs, labels = data
             inputs = inputs.to(device)
             labels = labels.to(device)

             optimizer.zero_grad()
             outputs = model(inputs)
             loss = criterion(outputs.to(device), labels)
             loss.backward()
             optimizer.step()
             temp_loss.append(loss.item())
        loss_history.append(np.sum(temp_loss)/len(temp_loss))
        scheduler.step()
        print(f'Epoch {epoch} / {epochs}: \
               avg. loss of last epoch {loss_history[-1]}')
    torch.save(model.state_dict(), "model_weights_x4.pt")
    print("Tests for first 100 cropped images")
    output_hr = model(lr_images[:100])

    print(psnr(output_hr, hr_images[:100]))

    plt.plot(loss_history)

    plt.savefig("loss_plot_1000_epoch_x4.png")


if __name__ == "__main__":
    main()
