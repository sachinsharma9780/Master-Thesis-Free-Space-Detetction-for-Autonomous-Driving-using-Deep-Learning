# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super(DownsamplerBlock, self).__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        #print('conv', self.conv(input).shape, 'pool', self.pool(input).shape, 'output', output.shape)
        output = self.bn(output)
        return F.relu(output)
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super(non_bottleneck_1d, self).__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super(Encoder, self).__init__()
        self.initial_block = DownsamplerBlock(3,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
           self.layers.append(non_bottleneck_1d(64, 0.03, 1)) 

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        # start of road classification layers

        self.road_layers = nn.ModuleList()

        self.road_layers.append(nn.Conv2d(128, 256, (3, 3), stride=2, padding=2))
        self.road_layers.append(nn.MaxPool2d(2, stride=2))
        self.road_layers.append(nn.BatchNorm2d(256, eps=1e-3))
        self.road_layers.append(non_bottleneck_1d(256, 0.3, 1))
        #self.road_layers.append(non_bottleneck_1d(256, 0.3, 1))

        self.road_layers.append(nn.Conv2d(256, 512, (3, 3), stride=2, padding=2))
        self.road_layers.append(nn.MaxPool2d(2, stride=2))
        self.road_layers.append(nn.BatchNorm2d(512, eps=1e-3))
        self.road_layers.append(non_bottleneck_1d(512, 0.3, 1))
        #self.road_layers.append(non_bottleneck_1d(512, 0.3, 1))

        self.road_linear_1 = nn.Linear(512 * 3 * 5, 1024)
        self.output_road = nn.Linear(1024, 4)

        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        bs = input.size()[0]
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        output_road = output
        for layer in self.road_layers:
            output_road = layer(output_road)

        output_road = output_road.view(bs, -1)
        output_road = self.road_linear_1(output_road)

        output_road = self.output_road(output_road)

        if predict:
            output = self.output_conv(output)

        return output, output_road


class EdgeMaps(nn.Module):
    def __init__(self):
        super(EdgeMaps, self).__init__()

        self.edge_layers = nn.ModuleList()
        self.conv1 = nn.Conv2d(1, 64, (3, 3), stride=2, padding=1)
        self.edge_layers.append(nn.MaxPool2d(2, stride=2))
        self.edge_layers.append(nn.BatchNorm2d(64, eps=1e-3))
        self.edge_layers.append(non_bottleneck_1d(64, 0.3, 1))

        self.edge_layers.append(nn.Conv2d(64, 128, (3, 3), stride=2, padding=1))
        #self.edge_layers.append(nn.MaxPool2d(2, stride=2))
        self.edge_layers.append(nn.BatchNorm2d(128, eps=1e-3))
        self.edge_layers.append(non_bottleneck_1d(128, 0.3, 1))


    def forward(self, input):
        output = self.conv1(input)

        for layer in self.edge_layers:
            output = layer(output)

        return output


"""class EdgeMaps(nn.Module):
    def __init__(self):
        super(EdgeMaps, self).__init__()
        self.edge_maps_layers = nn.ModuleList()

        self.edge_maps_initial_output = nn.Conv2d(3, 16, (3,3), stride=2, padding=2)
        self.edge_maps_layers.append(nn.MaxPool2d(2, stride=2))
        self.edge_maps_layers.append(nn.BatchNorm2d(16, eps=1e-3))
        self.edge_maps_layers.append(non_bottleneck_1d(16, 0.3, 1))

        self.edge_maps_layers.append(nn.Conv2d(16, 64, (3, 3), stride=2, padding=2))
        self.edge_maps_layers.append(nn.MaxPool2d(2, stride=2))
        self.edge_maps_layers.append(nn.BatchNorm2d(64, eps=1e-3))
        self.edge_maps_layers.append(non_bottleneck_1d(64, 0.3, 1))

        self.edge_maps_layers.append(nn.Conv2d(64, 128, (3, 3), stride=2, padding=2))
        self.edge_maps_layers.append(nn.MaxPool2d(2, stride=2))
        self.edge_maps_layers.append(nn.BatchNorm2d(128, eps=1e-3))
        self.edge_maps_layers.append(non_bottleneck_1d(128, 0.3, 1))

    def forward(self, input):
            output_edge = self.edge_maps_initial_output(input)
            for layer in self.edge_maps_layers:
                output_edge = layer(output_edge)

            return output_edge"""



class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super(UpsamplerBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class Decoder (nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


# perform fusion of feature maps from Encoder and Edge Maps
# elementwise addition
def fusion(encoder_features, edge_features): 
    #print('enc', encoder_features.shape, 'edge feature', edge_features.shape)
    # addition
    fused_output = torch.add(encoder_features, edge_features)
    # multiplication
    #fused_output = torch.mul(encoder_features, edge_features)
    # concatenation
    #fused_output = torch.cat((encoder_features, edge_features), 1)
    #print('fused shape', fused_output.shape)
    return fused_output


#ERFNet
class Net(nn.Module):
    def __init__(self, num_classes=3, encoder=None):  #use encoder to pass pretrained encoder
        super(Net, self).__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
            self.edge_maps = EdgeMaps()
        else:
            self.encoder = encoder
        
        self.decoder = Decoder(num_classes)


    def forward(self, input_3d, input_2d, only_encode=False):
        if only_encode:
            return self.encoder.forward(input_3d, predict=True)
        else:
            output, output_road = self.encoder(input_3d)    #predict=False by default
            #print('Shape of Encoder output: ', output.shape)
            edge_feature_maps = self.edge_maps(input_2d)
            # perform fusion
            fused_output = fusion(output, edge_feature_maps)
            return self.decoder.forward(fused_output), output_road





