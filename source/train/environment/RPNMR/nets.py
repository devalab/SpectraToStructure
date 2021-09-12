import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, init_std=1e-6, 
                 output_dim = None):
        #print("Creating resnet with input_dim={} hidden_dim={} depth={}".format(input_dim, hidden_dim, depth))
        assert(depth >= 0)
        super(ResNet, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        if output_dim is None:
            output_dim = hidden_dim

        self.linear_out = nn.Linear(hidden_dim, output_dim)
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim, init_std) for i in range(depth)])

    def forward(self, input):
        input = input.view(input.size(0), -1)
        x = self.linear_in(input)
        for res_block in self.res_blocks:
            x = res_block(x)
        return self.linear_out(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim, noise=1e-6):
        super(ResidualBlock, self).__init__()
        self.noise = noise
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim, bias = False)
        self.l1.bias.data.uniform_(-self.noise,self.noise)
        self.l1.weight.data.uniform_(-self.noise,self.noise) #?!
        self.l2.weight.data.uniform_(-self.noise,self.noise)

    def forward(self, x):
        return x + self.l2(F.relu(self.l1(x)))

    
class SumLayers(nn.Module):
    """
    Fully-connected layers that sum elements in a set
    """
    def __init__(self, input_D, input_max, filter_n, layer_count):
        super(SumLayers, self).__init__()
        self.fc1 = nn.Linear(input_D, filter_n)
        self.relu1 = nn.ReLU()
        self.fc_blocks = nn.ModuleList([nn.Sequential(nn.Linear(filter_n, filter_n), nn.ReLU()) for _ in range(layer_count-1)])
        
        
    def forward(self, X, present):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        
        xt = X  # .transpose(1, 2)
        x = self.fc1(xt) 
        x = self.relu1(x)
        for fcre in self.fc_blocks:
            x = fcre(x)
        
        x = (present.unsqueeze(-1) * x)
        
        return x.sum(1)

    

class ClusterCountingNetwork(nn.Module):
    """
    A network to count the number of points in each
    cluster. Very simple, mostly for pedagogy
    """
    
    def __init__(self, input_D, input_max, 
                 sum_fc_filternum, sum_fc_layercount, 
                 post_sum_fc, post_sum_layercount, 
                 output_dim):
        super(ClusterCountingNetwork, self).__init__()

        self.sum_layer = SumLayers(input_D, input_max, sum_fc_filternum, 
                                  sum_fc_layercount)
        self.post_sum = nn.Sequential(ResNet(sum_fc_filternum, 
                                             post_sum_fc,
                                             post_sum_layercount, 
                                         ),
                                   nn.Linear(post_sum_fc, output_dim), 
                                   nn.ReLU()
                                     )
    def forward(self, X, present):
        sum_vals = self.sum_layer(X, present)
        return self.post_sum(sum_vals)


class ResNetRegression(nn.Module):
    def __init__(self, D, block_sizes, INT_D, FINAL_D, 
                 use_batch_norm=False, OUT_DIM=1):
        super(ResNetRegression, self).__init__()

        layers = [nn.Linear(D, INT_D)]

        for block_size in block_sizes:
            layers.append(ResNet(INT_D, INT_D, block_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(INT_D))
        layers.append(nn.Linear(INT_D, FINAL_D))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(FINAL_D, OUT_DIM))
                
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X) 




class PyTorchResNet(nn.Module):
    """
    This is a modification of the default pytorch resnet to allow 
    for different input sizes, numbers of channels, kernel sizes, 
    and number of block layers and classes
    """

    def __init__(self, block, layers, input_img_size = 64, 
                 num_channels=3, 
                 num_classes=1, first_kern_size=7, 
                 final_avg_pool_size=7, inplanes=64):
        self.inplanes = inplanes
        super(PyTorchResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, self.inplanes, kernel_size=first_kern_size, 
                               stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block_layers = []
        for i, l in enumerate(layers):
            stride = 1 if i == 0 else 2 
            layer = self._make_layer(block, 64*2**i, l, stride=stride)
            self.block_layers.append(layer)

        self.block_layers_seq = nn.Sequential(*self.block_layers)

        last_image_size = input_img_size // (2**(len(layers) +1))
        post_pool_size = last_image_size - final_avg_pool_size+1
        self.avgpool = nn.AvgPool2d(final_avg_pool_size, stride=1, padding=0)
        expected_final_planes = 32 * 2**len(layers)

        self.fc = nn.Linear(expected_final_planes * block.expansion * post_pool_size**2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block_layers_seq(x)
        # for l in self.block_layers:
            
        #     x = l(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



class SimpleGraphModel(nn.Module):
    """
    Simple graph convolution model that outputs dense features post-relu

    Add final layer for regression or classification
    
    """
    def __init__(self, MAX_N, input_feature_n, 
                 output_features_n, noise=1e-5, 
                 single_out_row=True, batch_norm = False, input_batch_norm=False):
        super(SimpleGraphModel, self).__init__()
        self.MAX_N = MAX_N
        self.input_feature_n = input_feature_n
        self.output_features_n = output_features_n
        self.noise = noise
        self.linear_layers = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.use_batch_norm = batch_norm
        self.input_batch_norm = input_batch_norm
        if self.input_batch_norm:
            self.input_batch_norm_layer = nn.BatchNorm1d(input_feature_n)

        for i in range(len(output_features_n)):
            if i == 0:
                lin = nn.Linear(input_feature_n, output_features_n[i])
            else:
                lin = nn.Linear(output_features_n[i-1], output_features_n[i])
            lin.bias.data.uniform_(-self.noise,self.noise)
            lin.weight.data.uniform_(-self.noise,self.noise) #?!

            self.linear_layers.append(lin)
            self.relus.append(nn.ReLU())
            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(output_features_n[i] ))

        self.single_out_row = single_out_row

    def forward(self, args):
        (G, x, tgt_out_rows) = args
        if self.input_batch_norm:
            x = self.input_batch_norm_layer(x.reshape(-1, self.input_feature_n))
            x = x.reshape(-1, self.MAX_N, self.input_feature_n)
            


        for l in range(len(self.linear_layers)):
            x = self.linear_layers[l](x)
            x = torch.bmm(G, x)
            x = self.relus[l](x)
            if self.use_batch_norm:
                x = self.batch_norms[l](x.reshape(-1, self.output_features_n[l]))
                x = x.reshape(-1, self.MAX_N, self.output_features_n[l])
        if self.single_out_row:
            return torch.stack([x[i, j] for i, j in enumerate(tgt_out_rows)])
        else:
            return x
        

class ResGraphModel(nn.Module):
    """
    Graphical resnet with batch norm structure
    """
    def __init__(self, MAX_N, input_feature_n, 
                 output_features_n, noise=1e-5, 
                 single_out_row=True, batch_norm = False, 
                 input_batch_norm=False, resnet=True):
        super(ResGraphModel, self).__init__()
        self.MAX_N = MAX_N
        self.input_feature_n = input_feature_n
        self.output_features_n = output_features_n
        self.noise = noise
        self.linear_layers = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.use_batch_norm = batch_norm
        self.input_batch_norm = input_batch_norm
        self.use_resnet = resnet

        if self.input_batch_norm:
            self.input_batch_norm_layer = nn.BatchNorm1d(input_feature_n)

        for i in range(len(output_features_n)):
            if i == 0:
                lin = nn.Linear(input_feature_n, output_features_n[i])
            else:
                lin = nn.Linear(output_features_n[i-1], output_features_n[i])
            #nn.init.kaiming_uniform_(lin.weight.data, nonlinearity='relu')
            #nn.init.kaiming_uniform_(lin.weight.data, nonlinearity='relu')

            lin.bias.data.uniform_(-self.noise,self.noise)
            lin.weight.data.uniform_(-self.noise,self.noise) #?!

            self.linear_layers.append(lin)
            self.relus.append(nn.ReLU())
            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(output_features_n[i] ))

        self.single_out_row = single_out_row

    def forward(self, args):
        (G, x, tgt_out_rows) = args
        if self.input_batch_norm:
            x = self.input_batch_norm_layer(x.reshape(-1, self.input_feature_n))
            x = x.reshape(-1, self.MAX_N, self.input_feature_n)
            


        for l in range(len(self.linear_layers)):
            x1 = torch.bmm(G, self.linear_layers[l](x))
            x2 = self.relus[l](x1)
            
            if x.shape == x2.shape and self.use_resnet:
                x3 = x2 + x
            else:
                x3 = x2
            if self.use_batch_norm:
                x = self.batch_norms[l](x3.reshape(-1, self.output_features_n[l]))
                x = x.reshape(-1, self.MAX_N, self.output_features_n[l])
            else:
                x = x3
        if self.single_out_row:
            return torch.stack([x[i, j] for i, j in enumerate(tgt_out_rows)])
        else:
            return x
        

        

def goodmax(x, dim):
    return torch.max(x, dim=dim)[0]

class GraphMatLayer(nn.Module):
    def __init__(self, C, P , GS=1,  
                 noise=1e-6, agg_func=None, 
                 dropout=0.0):
        """
        Pairwise layer -- takes a N x M x M x C matrix
        and turns it into a N x M x M x P matrix after
        multiplying with a graph matrix N x M x M
        
        if GS != 1 then there will be a per-graph-channel 
        linear layer
        """
        super(GraphMatLayer, self).__init__()

        self.GS = GS
        self.noise=noise

        self.linlayers = nn.ModuleList()
        self.dropout = dropout
        self.dropout_layers = nn.ModuleList()
        for ll in range(GS):
            l = nn.Linear(C, P)
            l.bias.data.normal_(0.0, self.noise)
            l.weight.data.normal_(0.0, self.noise) #?!
            self.linlayers.append(l)
            if dropout > 0.0:
                self.dropout_layers.append(nn.Dropout(p=dropout))
            
        #self.r = nn.PReLU()
        self.r = nn.ReLU()
        self.agg_func = agg_func
 
    def forward(self, G, x):
        def apply_ll(i, x):
            y = self.linlayers[i](x)
            if self.dropout > 0:
                y = self.dropout_layers[i](y)
            return y

        multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)])
        # this is per-batch-element
        xout = torch.stack([torch.matmul(G[i], multi_x[:, i]) for i in range(x.shape[0])])

        x = self.r(xout)
        if self.agg_func is not None:
            x = self.agg_func(x, dim=1)
        return x
        
class GraphMatLayers(nn.Module):
    def __init__(self, input_feature_n, 
                 output_features_n, resnet=False, GS=1, 
                 noise=1e-5, agg_func=None, dropout=0.0):
        super(GraphMatLayers, self).__init__()
        
        self.gl = nn.ModuleList()
        self.resnet = resnet

        for li in range(len(output_features_n)):
            if li == 0:
                gl = GraphMatLayer(input_feature_n, output_features_n[0],
                                   noise=noise, agg_func=agg_func, GS=GS, 
                                   dropout=dropout)
            else:
                gl = GraphMatLayer(output_features_n[li-1], 
                                   output_features_n[li], 
                                   noise=noise, agg_func=agg_func, GS=GS, 
                                   dropout=dropout)
            
            self.gl.append(gl)
    def forward(self, G, x):
        for gl in self.gl:
            x2 = gl(G, x)
            if self.resnet:
                if x.shape == x2.shape:
                    x3 = x2 + x
                else:
                    x3 = x2
            else:
                x3 = x2
            x = x3
        

        return x

class GraphMatHighwayLayers(nn.Module):
    def __init__(self, input_feature_n, 
                 output_features_n, resnet=False, GS=1, 
                 noise=1e-5, agg_func=None):
        super(GraphMatHighwayLayers, self).__init__()
        
        self.gl = nn.ModuleList()
        self.resnet = resnet

        for li in range(len(output_features_n)):
            if li == 0:
                gl = GraphMatLayer(input_feature_n, output_features_n[0],
                                   noise=noise, agg_func=agg_func, GS=GS)
            else:
                gl = GraphMatLayer(output_features_n[li-1], 
                                   output_features_n[li], 
                                   noise=noise, agg_func=agg_func, GS=GS)
            
            self.gl.append(gl)

    def forward(self, G, x):
        highway_out = []
        for gl in self.gl:
            x2 = gl(G, x)
            if self.resnet:
                if x.shape == x2.shape:
                    x3 = x2 + x
                else:
                    x3 = x2
            else:
                x3 = x2
            x = x3
            highway_out.append(x2)

        return x, torch.stack(highway_out, -1)


def batch_diagonal_extract(x):
    BATCH_N, M, _, N = x.shape

    return torch.stack([x[:, i, i, :] for i in range(M)], dim=1)

class GraphMatModel(nn.Module):
    def __init__(self, g_feature_n, g_feature_out_n, resnet=True, 
                 noise=1e-5, GS=1, OUT_DIM=1 ):
        """
        g_features_in : how many per-edge features
        g_features_out : how many per-edge features 
        """
        super(GraphMatModel, self).__init__()

        self.gml = GraphMatLayers(g_feature_n, g_feature_out_n, 
                                  resnet=resnet, noise=noise, GS=GS)
        
    
        self.lin_out = nn.Linear(g_feature_out_n[-1], OUT_DIM )
        #torch.nn.init.kaiming_uniform_(self.lin_out.weight.data, nonlinearity='relu')

    def forward(self, args):
        (G, x_G) = args

        G_features = self.gml(G, x_G)
        ## OLD WAY
        
        g_diag = batch_diagonal_extract(G_features)
        x_1 = self.lin_out(g_diag)
        
        return x_1


class GraphVertModel(nn.Module):
    def __init__(self, g_feature_n, g_feature_out_n=None, 
                 int_d = None, layer_n = None, 

                 resnet=True, 
                 init_noise=1e-5, agg_func=None, GS=1, OUT_DIM=1, 
                 batch_norm=False, out_std= False, 
                 resnet_out = False, resnet_blocks = (3, ), 
                 resnet_d = 128, 
                 graph_dropout=0.0, 
                 force_lin_init=False):
        
        """

        
        """
        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n


        super(GraphVertModel, self).__init__()
        self.gml = GraphMatLayers(g_feature_n, g_feature_out_n, 
                                  resnet=resnet, noise=init_noise, agg_func=parse_agg_func(agg_func), 
                                  GS=GS, dropout=graph_dropout)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(g_feature_n)
        else:
            self.batch_norm = None

        self.resnet_out = resnet_out
        if not resnet_out:
            self.lin_out = nn.Linear(g_feature_out_n[-1], OUT_DIM)
        else:
            self.lin_out = ResNetRegression(g_feature_out_n[-1], 
                                                block_sizes = resnet_blocks, 
                                                INT_D = resnet_d, 
                                                FINAL_D=resnet_d, 
                                                OUT_DIM=OUT_DIM)

        self.out_std = out_std

        if out_std:
            self.lin_out_std1 = nn.Linear(g_feature_out_n[-1], 128)
            self.lin_out_std2 = nn.Linear(128, OUT_DIM)

            # self.lin_out_std = ResNetRegression(g_feature_out_n[-1], 
            #                                     block_sizes = resnet_blocks, 
            #                                     INT_D = resnet_d, 
            #                                     FINAL_D=resnet_d, 
            #                                     OUT_DIM=OUT_DIM)

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, init_noise)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, args, return_g_features = False):
        (G, x_G) = args
        
        BATCH_N, MAX_N, F_N = x_G.shape

        if self.batch_norm is not None:
            x_G_flat = x_G.reshape(BATCH_N*MAX_N, F_N)
            x_G_out_flat = self.batch_norm(x_G_flat)
            x_G = x_G_out_flat.reshape(BATCH_N, MAX_N, F_N)
        
        G_features = self.gml(G, x_G)
        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)
        g_squeeze_flat = g_squeeze.reshape(-1, G_features.shape[-1])
        
        if self.resnet_out:
            x_1 = self.lin_out(g_squeeze_flat).reshape(BATCH_N, MAX_N, -1)
        else:
            x_1 = self.lin_out(g_squeeze)
        if self.out_std:

            x_std = F.relu(self.lin_out_std1(g_squeeze))
            x_1_std = F.relu(self.lin_out_std2(x_std))
                           
            # g_2 = F.relu(self.lin_out_std(g_squeeze_flat))

            # x_1_std = g_2.reshape(BATCH_N, MAX_N, -1)

            return {'mu' : x_1, 'std' : x_1_std}
        else:
            return x_1


class GraphVertResOutModel(nn.Module):
    def __init__(self, g_feature_n, g_feature_out_n, resnet=True, 
                 noise=1e-5, agg_func=None, GS=1, OUT_DIM=1, 
                 batch_norm=False, out_std= False, 
                 force_lin_init=False):
        
        """

        
        """
        super(GraphVertResOutModel, self).__init__()
        self.gml = GraphMatLayers(g_feature_n, g_feature_out_n, 
                                  resnet=resnet, noise=noise, agg_func=agg_func, 
                                  GS=GS)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(g_feature_n)
        else:
            self.batch_norm = None

        print("g_feature_out_n[-1]=", g_feature_out_n[-1])

        self.lin_out = ResNetRegression(g_feature_out_n[-1], 
                                        block_sizes = [3], 
                                        INT_D = 128, 
                                        FINAL_D=1024, 
                                        OUT_DIM=OUT_DIM)

        self.out_std = out_std

        # if out_std:
        #     self.lin_out_std = nn.Linear(g_feature_out_n[-1], 32)
        #     self.lin_out_std1 = nn.Linear(32, OUT_DIM)

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, noise)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, args):
        (G, x_G) = args
        
        BATCH_N, MAX_N, F_N = x_G.shape

        if self.batch_norm is not None:
            x_G_flat = x_G.reshape(BATCH_N*MAX_N, F_N)
            x_G_out_flat = self.batch_norm(x_G_flat)
            x_G = x_G_out_flat.reshape(BATCH_N, MAX_N, F_N)
        
        G_features = self.gml(G, x_G)
        
        g_squeeze = G_features.squeeze(1).reshape(-1, G_features.shape[-1])


        x_1 = self.lin_out(g_squeeze)

        return x_1.reshape(BATCH_N, MAX_N, -1)

        # if self.out_std:
        #     x_1_std = F.relu(self.lin_out_std(g_squeeze))
        #     x_1_std = F.relu(self.lin_out_std1(x_1_std))

        #     return x_1, x_1_std
        # else:
        #     return x_1



def parse_agg_func(agg_func):
    if isinstance(agg_func, str):
        if agg_func == 'goodmax':
            return goodmax
        elif agg_func == 'sum':
            return torch.sum
        elif agg_func == 'mean':
            return torch.mean
        else:
            raise NotImplementedError()
    return agg_func

class GraphVertExtraLinModel(nn.Module):
    def __init__(self, g_feature_n, g_feature_out_n=None,
                 resnet=True, 
                 int_d = None, layer_n = None, 
                 init_noise=1e-5, agg_func=None, GS=1, 
                 combine_in = False, 
                 OUT_DIM=1,force_lin_init=False, 
                 use_highway = False, 
                 use_graph_conv = True, 
                 extra_lin_int_d = 128 ):
        
        """

        
        """
        super(GraphVertExtraLinModel, self).__init__()

        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n
        print("g_feature_n=", g_feature_n)
        self.use_highway = use_highway
        if use_highway:

            self.gml = GraphMatHighwayLayers(g_feature_n, g_feature_out_n, 
                                      resnet=resnet, noise=init_noise, 
                                      agg_func=parse_agg_func(agg_func), 
                                      GS=GS)
        else:

            self.gml = GraphMatLayers(g_feature_n, g_feature_out_n, 
                                      resnet=resnet, noise=init_noise, 
                                      agg_func=parse_agg_func(agg_func), 
                                      GS=GS)

        self.combine_in = combine_in
        self.use_graph_conv = use_graph_conv
        lin_layer_feat = 0
        if use_graph_conv:
            lin_layer_feat += g_feature_out_n[-1]

        if combine_in :
            lin_layer_feat += g_feature_n
        if self.use_highway:
            lin_layer_feat += np.sum(g_feature_out_n)
            
        self.lin_out1 = nn.Linear(lin_layer_feat, extra_lin_int_d)
        self.lin_out2 = nn.Linear(extra_lin_int_d, OUT_DIM)

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, init_noise)
                    nn.init.constant_(m.bias, 0)

    def forward(self, args):
        G = args[0]
        x_G = args[1]
        if self.use_highway:
            G_features, G_highway = self.gml(G, x_G)
            G_highway_flatten = G_highway.reshape(G_highway.shape[0], 
                                                  G_highway.shape[1], -1)
        else:
            G_features = self.gml(G, x_G)

        
        g_squeeze = G_features.squeeze(1)
        out_feat = []
        if self.use_graph_conv :
            out_feat.append(g_squeeze)
        if self.combine_in:
            out_feat.append(x_G)
        if self.use_highway:
            out_feat.append(G_highway_flatten)

        lin_input = torch.cat(out_feat, -1)
            
        x_1 = self.lin_out1(lin_input)
        x_2 = self.lin_out2(F.relu(x_1))
        
        return x_2


class MSELogNormalLoss(nn.Module):
    def __init__(self, use_std_term = True, 
                 use_log1p=True, std_regularize=0.0, 
                 std_pow = 2.0):
        super(MSELogNormalLoss, self).__init__()
        self.use_std_term = use_std_term
        self.use_log1p = use_log1p
        self.std_regularize = std_regularize
        self.std_pow = std_pow

    def __call__(self, y, mu, std):
        if self.use_log1p:
            log = torch.log1p
        else:
            log = torch.log
        std = std + self.std_regularize

        std_term = -0.5 * log(2*np.pi * std**2 ) 
        log_pdf = - (y-mu)**2/(2.0 * std **self.std_pow)
        if self.use_std_term :
            log_pdf += std_term 

        return -log_pdf.mean()


def log_normal_nolog(y, mu, std):
    element_wise =  -(y - mu)**2 / (2*std**2)  - std
    return element_wise

def log_student_t(y, mu, std, v=1.0):
    return -torch.log(1.0 + (y-mu)**2/(v * std)) - std

def log_normal(y, mu, std):
    element_wise =  -(y - mu)**2 / (2*std**2)  - torch.log(std)
    return element_wise

class MSECustomLoss(nn.Module):
    def __init__(self, use_std_term = True, 
                 use_log1p=True, std_regularize=0.0, 
                 std_pow = 2.0):
        super(MSECustomLoss, self).__init__()
        self.use_std_term = use_std_term
        self.use_log1p = use_log1p
        self.std_regularize = std_regularize
        self.std_pow = std_pow

    def __call__(self, y, mu, std):
        if self.use_log1p:
            log = torch.log1p
        else:
            log = torch.log
        std = std + self.std_regularize

        # std_term = -0.5 * log(2*np.pi * std**self.std_pow ) 
        # log_pdf = - (y-mu)**2/(2.0 * std **self.std_pow)

        # if self.use_std_term :
        #     log_pdf += std_term 

        # return -log_pdf.mean()
        return -log_normal(y, mu, std).mean()

class MaskedMSELoss(nn.Module):
    """
    Masked mean squared error
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mseloss = nn.MSELoss()
        
    def __call__(self, y, x, mask):
        x_masked = x[mask>0].reshape(-1, 1)
        y_masked = y[mask>0].reshape(-1, 1)
        return self.mseloss(x_masked, y_masked)

class MaskedMSSELoss(nn.Module):
    """
    Masked mean squared error
    """

    def __init__(self):
        super(MaskedMSSELoss, self).__init__()
        
    def __call__(self, y, x, mask):
        x_masked = x[mask>0].reshape(-1, 1)
        y_masked = y[mask>0].reshape(-1, 1)
        return ((x_masked - y_masked)**4).mean()



class MaskedMSEScaledLoss(nn.Module):
    """
    Masked mean squared error
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mseloss = nn.MSELoss()
        
    def __call__(self, y, x, mask):
        x_masked = x[mask>0].reshape(-1, 1)
        y_masked = y[mask>0].reshape(-1, 1)
        return self.mseloss(x_masked, y_masked)


class NormUncertainLoss(nn.Module):
    """
    Masked uncertainty loss
    """
    def __init__(self, 
                 mu_scale = torch.Tensor([1.0]), 
                 std_scale = torch.Tensor([1.0]), 
                 use_std_term = True, 
                 use_log1p=False, std_regularize=0.0, 
                 std_pow = 2.0):
        super(NormUncertainLoss, self).__init__()
        self.use_std_term = use_std_term
        self.use_log1p = use_log1p
        self.std_regularize = std_regularize
        self.std_pow = std_pow
        self.mu_scale = mu_scale
        self.std_scale = std_scale

    def __call__(self, pred, y,  mask):
        ### NOTE pred is a tuple! 
        mu, std = pred['mu'], pred['std']

        if self.use_log1p:
            log = torch.log1p
        else:
            log = torch.log
        std = std + self.std_regularize


        y_scaled = y / self.mu_scale
        mu_scaled = mu / self.mu_scale
        std_scaled = std / self.std_scale 

        y_scaled_masked = y_scaled[mask>0].reshape(-1, 1)
        mu_scaled_masked = mu_scaled[mask>0].reshape(-1, 1)
        std_scaled_masked = std_scaled[mask>0].reshape(-1, 1)
        # return -log_normal_nolog(y_scaled_masked, 
        #                          mu_scaled_masked, 
        #                          std_scaled_masked).mean()
        return -log_normal_nolog(y_scaled_masked, 
                              mu_scaled_masked, 
                              std_scaled_masked).mean()



