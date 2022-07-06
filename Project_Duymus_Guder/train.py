import os
import torch
import torch.nn as nn
import torch.optim as optim
import cifar
import model

def train(model, lr, bs, num_epochs):
    trainSet = cifar.loadCIFAR10(train=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloader = torch.utils.data.DataLoader(trainSet, batch_size=bs, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-5)

    loss_history = list()

    for epoch in range(num_epochs):
        acc_loss = 0.0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            acc_loss = loss.item() * inputs.size(0)

        epoch_loss = acc_loss / len(dataloader.dataset)
        loss_history.append(epoch_loss)

        print(f"Epoch {epoch+1} / {num_epochs} loss: {epoch_loss}")

    if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')

    torch.save(model.state_dict(), "checkpoints/checkpoint.pt")

    return loss_history

if __name__ == '__main__':
    model = model.VGG16()
    losses = train(model, lr=1e-3, bs=32, num_epochs=20)