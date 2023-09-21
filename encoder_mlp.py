from matplotlib.pyplot import axes
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import gc

class MLP(nn.Module):

    """
    MLP class: contains the model implemenetation for the MLP module used as the encoder
    """

    def __init__(self, args):
        
        

        super(MLP, self).__init__()
        self.feature_dim = args.features_dim
        self.embed_dim = args.num_f_maps
        self.num_clusters = args.num_classes

        
        self.fc1 = nn.Linear(self.feature_dim, self.embed_dim * 2)
        self.fc2 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.prototype_layer = nn.Linear(self.embed_dim, self.num_clusters, bias=False)

        
        self._init_weights()

    def forward(self, x , mask):
        # x = self.resnet(x).view(x.shape[0], -1)
        #print(" shape of x: " , x.shape)
        #x= x[0]
        x= torch.permute(x, (0,2,1))

        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        embs = x.clone()
        proto_scores = self.prototype_layer(x)
        #print(proto_scores.shape)
        proto_scores = proto_scores.reshape((proto_scores.size(0) * proto_scores.size(1), proto_scores.size(2)))
        #embs = embs.reshape((embs.size(0) * embs.size(1), embs.size(2)))
        proto_scores= [[proto_scores]]
        embs =[torch.permute(embs, (0,2,1))]
        
        #proto_scores =[torch.permute(proto_scores, (0,2,1))]
        return proto_scores, embs

    def get_embeddings(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        x = x.reshape((x.size(0) * x.size(1), x.size(2)))

        return x

    def get_prototypes(self):
        return self.prototype_layer.weight.data.clone()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if type(m.bias) != type(None):
                    nn.init.constant_(m.bias, 0)