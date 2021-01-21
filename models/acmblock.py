import torch
import torch.nn as nn
import torch.nn.functional as F

class ACMBlock(nn.Module):
    def __init__(self, in_channels):
        super(ACMBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.k_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, (1,1), groups=64),
        )

        self.q_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, (1,1), groups=64),
        )

        self.global_pooling = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels//2, (1,1)),
            nn.ReLU(),
            nn.Conv2d(self.out_channels//2, self.out_channels, (1,1)),
            nn.Sigmoid()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.normalize = nn.Softmax(dim=3)

    def _get_normalized_features(self, x):
        ''' Get mean vector by channel axis '''
        
        c_mean = self.avgpool(x)
        return c_mean

    def get_K(self, x):
        K = self.k_conv(x)

        b, c, h, w = K.shape

        K = K.view(b, c, 1, h*w)
        K = self.normalize(K)
        K = K.view(b, c, h, w)

        K = torch.einsum('nchw,nchw->nc',[K, x])

        return K

    def get_Q(self, x):
        Q = self.q_conv(x)

        b, c, h, w = Q.shape

        Q = Q.view(b, c, 1, h*w)
        Q = self.normalize(Q)
        Q = Q.view(b, c, h, w)

        Q = torch.einsum('nchw,nchw->nc',[Q, x])

        return Q
    
    def get_channel_weights(self, x):
        mean_features = self._get_normalized_features(x)
        normalized_features = x - mean_features
        
        channel_weights = self.global_pooling(mean_features)

        return channel_weights
    
    def get_orth_loss(self, k, q):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        orth_loss = cos(k, q)
        orth_loss = torch.mean(orth_loss, dim=0)
        return orth_loss

    def forward(self, x1, x2):
        K = self.k_conv(x1)
        Q = self.q_conv(x2)

        b, c, h, w = K.shape

        K = K.view(b, c, 1, h*w)
        K = self.normalize(K)
        K = K.view(b, c, h, w)

        Q = Q.view(b, c, 1, h*w)
        Q = self.normalize(Q)
        Q = Q.view(b, c, h, w)

        K = torch.einsum('nchw,nchw->nc',[K, x1])
        Q = torch.einsum('nchw,nchw->nc',[Q, x2])
        K = K.view(K.shape[0], K.shape[1], 1, 1)
        Q = Q.view(Q.shape[0], Q.shape[1], 1, 1)

        mean_features1 = self._get_normalized_features(x1)
        mean_features2 = self._get_normalized_features(x2)
        channel_weights1 = self.global_pooling(mean_features1)
        channel_weights2 = self.global_pooling(mean_features2)
        
        out1 = x1 + (K - Q)
        out2 = x2 + (K - Q)
        
        out1 = channel_weights1 * out1
        out2 = channel_weights2 * out2
        
        orth_loss = self.get_orth_loss(K,Q)

        return out1, out2, orth_loss

### Test
if __name__ == '__main__':
    test_vector = torch.randn((1,2,3,3), dtype=torch.float32)
    acm = ACMBlock(in_channels=test_vector.shape[1])

    result = acm(test_vector)
