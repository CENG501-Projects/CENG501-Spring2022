import os
import torch
import torch.nn as nn
import torch.optim as optim
import cifar
import model

def train(model, lr, bs, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    trainSet = cifar.loadCIFAR10(train=True)
    dataloader = torch.utils.data.DataLoader(trainSet, batch_size=bs, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    loss_history = list()

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        loss_history.append(epoch_loss)

        print(f"Epoch {epoch+1} / {num_epochs} loss: {epoch_loss}, accuracy: {epoch_acc}")

    if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')

    torch.save(model.state_dict(), f"checkpoints/checkpoint.pt")

    return loss_history

if __name__ == '__main__':
    lr = 1e-2
    bs = 256
    epochs = 50

    torch.manual_seed(501)

    model = model.VGG16()
    losses = train(model, lr=lr, bs=bs, num_epochs=epochs)