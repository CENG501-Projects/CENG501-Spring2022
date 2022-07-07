import torch
import torch.nn as nn
import torch.optim as optim
import cifar
import model

def test(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    testSet = cifar.loadCIFAR10(train=False)
    dataloader = torch.utils.data.DataLoader(testSet, batch_size=128)   # to avoid CUDA out of mem. errors
    corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        corrects += torch.sum(preds == labels.data)

    print("Test set accuracy: {:.3f}%".format(100 * corrects / len(dataloader.dataset)))

if __name__ == '__main__':
    model = model.VGG16()
    model.load_state_dict(torch.load('checkpoints/checkpoint.pt'))
    model.eval()

    with torch.no_grad():
        test(model)