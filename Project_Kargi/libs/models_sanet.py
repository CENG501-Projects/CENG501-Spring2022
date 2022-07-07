
import torch
import torch.nn as nn
from torchvision import transforms


def calc_mean_std(feat, eps = 1e-5):
    """ Calculate mean and std. Epsilon is added to the variance to avoid divide-by-zero """
    size = feat.size()
    
    assert(len(size) == 4) # image should be 4D
    N, C = size[:2]
    feat_var = feat.view(N,C,-1).var(dim=2) + eps # reshape it into shape (N,C,HxW) and calculate variance for all N,C
    feat_std = feat_var.sqrt().view(N,C,1,1)
    feat_mean = feat.view(N,C,-1).mean(dim=2).view(N,C,1,1)
    return feat_mean,feat_std


def mean_variance_norm(feat):
    """ Normalize mean and variance """
    size = feat.size()

    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size) # make sure that all operations are vectorized
                                                                    # N,C,1,1 -> N,C,H,W   
    return normalized_feat

def _calc_feat_flatten_mean_std(feat):
    """  Calculate mean and std of an 3D view """
    size = feat.size()
    assert(len(size) == 3)
    assert(isinstance(feat,torch.FloatTensor))
    feat_flatten = feat.view(3,-1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std

def weighted_mse_loss(input,target,weights = None):
  assert input.size() == target.size()
  size = input.size()
  if weights == None:
    weights = torch.ones(size = size[0])

  if len(size) == 3: # gram matrix is B,C,C
    se = ((input.view(size[0],-1) - target.view(size[0],-1))**2)
    return (se.mean(dim = 1)*weights).mean()


def gram_matrix_(input, normalize =True):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    v = input.view(a,b, c * d)  # resise F_XL into \hat F_XL
    vt = v.permute(0,2,1)
    # C,HxW , HxW, C
    G = v.matmul(vt)  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    if normalize:
      return G.div(b * c * d)
    else:
      return G

def gram_matrix(x, normalize=True):
    '''
    Generate gram matrices of the representations of content and style images.
    '''
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if normalize:
        gram /= ch * h * w
    return gram

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class SANet(nn.Module):
    
    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim = -1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))
        
    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        S = self.sm(S)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        return O

class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = SANet(in_planes = in_planes)
        self.sanet5_1 = SANet(in_planes = in_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1):
        s41 = self.sanet4_1(content4_1, style4_1)
        s51 = self.sanet5_1(content5_1, style5_1)

        su51 = self.upsample5_1(s51)

        s41_size = s41.size()
        su51_size = su51.size()

        #print(s41.size(), su51.size(), s51.size())
        if(s41_size[2] != su51_size[2]):
            s41 = s41[:,:,:min(s41_size[2],su51_size[2]),:]
            su51 = su51[:,:,:min(s41_size[2],su51_size[2]),:]
        if s41_size[3] != su51_size[3]:
            s41 = s41[:,:,:,:min(s41_size[3],su51_size[3])]
            su51 = su51[:,:,:,:min(s41_size[3],su51_size[3])]

        #print(s41.size(), su51.size(), s51.size())
        assert s41.size() == su51.size()
        x = s41 + su51
        x = self.merge_conv_pad(x)
        return self.merge_conv(x)

def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def calc_content_loss(input, target, norm = False):
    mse_loss = nn.MSELoss()
    if(norm == False):
        return mse_loss(input, target)
    else:
        return mse_loss(mean_variance_norm(input), mean_variance_norm(target))
        
def calc_ast_style_loss_normalized(input, target):
    G1 = gram_matrix(input, False)
    G2 = gram_matrix(target, False).detach() # we dont need the gradient of the target

    size = input.size()
    assert(len(size) == 4)

    g1_norm = torch.linalg.norm(G1,dim = (1,2))
    g2_norm = torch.linalg.norm(G2,dim = (1,2))

    size = G1.size()
    Nl = size[1] * size[2] # Or C x C = C^2
    normalize_term =  (torch.square(g1_norm) + torch.square(g2_norm))/Nl  #

    weights = (1/normalize_term)
    #weights = weights.view(size[0],1)
    return weighted_mse_loss(G1,G2,weights)

def return_normalize_weight(input, target):
    G1 = gram_matrix(input, False)
    G2 = gram_matrix(target, False).detach() # we dont need the gradient of the target

    size = input.size()
    assert(len(size) == 4)

    g1_norm = torch.linalg.norm(G1,dim = (1,2))
    g2_norm = torch.linalg.norm(G2,dim = (1,2))

    size = G1.size()
    Nl = size[1] * size[2] # Or C x C = C^2
    normalize_term =  (torch.square(g1_norm) + torch.square(g2_norm))/Nl  #supremum

    return normalize_term

def calc_ast_style_loss(input, target):
    """
    Interestingly, authors assumes that calculation of G1 and G2 is not normalized
    This can create an issue?. For this reason, let us test for
        1- Normalized
        2- Unnormalized
        3- Newly defined Normalization term
    """
    mse_loss = nn.MSELoss()
    G1 = gram_matrix(input)
    G2 = gram_matrix(target).detach() # we dont need the gradient of the target

    size = input.size()
    assert(len(size) == 4)
    assert G1.size() == G2.size()
    
    return mse_loss(G1,G2)

def calc_ast_style_loss_unnormalized(input, target):
    """
    Interestingly, authors assumes that calculation of G1 and G2 is not normalized
    This can create an issue?. For this reason, let us test for
        1- Normalized
        2- Unnormalized
        3- Newly defined Normalization term
    """
    mse_loss = nn.MSELoss()
    G1 = gram_matrix(input, False)
    G2 = gram_matrix(target, False).detach() # we dont need the gradient of the target

    size = input.size()
    assert(len(size) == 4)
    assert G1.size() == G2.size()
    
    return mse_loss(G1,G2)

class Net(nn.Module):
    def __init__(self, encoder, decoder, start_iter):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())

        ##############################################################
        # b1r2, b2r2, b3r3, b4r4 for style encoding     
        # b3r3 for content loss
        ##############################################################

        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        #transform
        self.transform = Transform(in_planes = 512)
        self.decoder = decoder
        if(start_iter > 0):
            self.transform.load_state_dict(torch.load('transformer_iter_' + str(start_iter) + '.pth'))
            self.decoder.load_state_dict(torch.load('decoder_iter_' + str(start_iter) + '.pth'))
        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        """
        Basically, traverse all the intermediate layers and add the last output of it to the results. 
        """
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target, norm = False):
        if(norm == False):
          return self.mse_loss(input, target)
        else:
          return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_ast_style_loss_normalized(self, input, target):
        G1 = gram_matrix(input, False)
        G2 = gram_matrix(target, False).detach() # we dont need the gradient of the target

        size = input.size()
        assert(len(size) == 4)

        g1_norm = torch.linalg.norm(G1,dim = (1,2))
        g2_norm = torch.linalg.norm(G2,dim = (1,2))

        size = G1.size()
        Nl = size[1] * size[2] # Or C x C = C^2
        normalize_term =  (torch.square(g1_norm) + torch.square(g2_norm))/Nl  #

        weights = (1/normalize_term)
        #weights = weights.view(size[0],1)
        return weighted_mse_loss(G1,G2,weights)
      
    def calc_ast_style_loss(self, input, target):
        """
        Interestingly, authors assumes that calculation of G1 and G2 is not normalized
        This can create an issue?. For this reason, let us test for
          1- Normalized
          2- Unnormalized
          3- Newly defined Normalization term
        """
        G1 = gram_matrix(input)
        G2 = gram_matrix(target).detach() # we dont need the gradient of the target

        size = input.size()
        assert(len(size) == 4)

        return self.mse_loss(G1,G2)

    def calc_style_loss(self, input, target):
        
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
        
        #return self.calc_ast_style_loss_normalized(input,target)

    def forward(self, content, style):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        stylized = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
        g_t = self.decoder(stylized)
        g_t_feats = self.encode_with_intermediate(g_t)
        loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3], norm = True) + self.calc_content_loss(g_t_feats[4], content_feats[4], norm = True)
        
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        """IDENTITY LOSSES"""
        Icc = self.decoder(self.transform(content_feats[3], content_feats[3], content_feats[4], content_feats[4]))
        Iss = self.decoder(self.transform(style_feats[3], style_feats[3], style_feats[4], style_feats[4]))
        l_identity1 = self.calc_content_loss(Icc, content) + self.calc_content_loss(Iss, style)
        Fcc = self.encode_with_intermediate(Icc)
        Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i], style_feats[i])
        return loss_c, loss_s, l_identity1, l_identity2
        