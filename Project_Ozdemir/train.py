import torch


def train(model, criterion, optimizer, epochs, data):

	model.train()

	edge_index, x, shuffled_x = data.edge_index, data.x, data.shuffled_x

	loss_history = []

	for epoch in range(epochs):

		optimizer.zero_grad()

		disc_out = model(edge_index, x, shuffled_x)

		lbl = torch.ones(x.shape[0])
		shuffled_lbl = torch.zeros(x.shape[0])
		conc_lbl = torch.cat((lbl, shuffled_lbl), 0)
		loss = criterion(disc_out, conc_lbl)

		loss.backward()
		optimizer.step()

		loss_history.append(loss.item())

	return loss_history