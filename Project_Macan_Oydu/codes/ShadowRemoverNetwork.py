import torch
import torch.nn as nn
from NetworkA import NetworkA_iter
from NetworkInit import NetworkInit

class ShadowRemoverNetwork(nn.Module):

    def __init__(self, Eta = 0.01, Beta = 0.01, Lambda = 0.01, K = 4):
        super(ShadowRemoverNetwork, self).__init__()
        self.K = K
        self.networkA = NetworkA_iter(Eta=Eta, Beta=Beta, Lambda=Lambda)
        self.networkInit = NetworkInit()

    def forward(self, I_s, M):
        I_ns_history = []
        A_history = []
        I_ns, A = self.networkInit(torch.cat((I_s,M),dim=1))
        I_ns_history.append(I_ns)
        A_history.append(A)
        for _ in range(self.K):
            I_ns, A = self.networkA(M,A,I_ns,I_s)
            I_ns_history.append(I_ns)
            A_history.append(A)
        return I_ns_history, A_history

if __name__ == "__main__":
    model = ShadowRemoverNetwork()
    input = torch.randn(1, 3, 256, 256)
    mask = torch.randn(1, 1, 256, 256)
    I_ns, A = model(input,mask)
    print('-'*50)
    print(I_ns[-1].shape)
    print('#generator parameters:', sum(param.numel() for param in model.parameters()))