from torch import nn


def dwise_conv(ch_in, stride=1):
    return (
        nn.Sequential(
            #depthwise
            nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, stride=stride, groups=ch_in, bias=False),
            nn.InstanceNorm2d(ch_in),
            nn.ReLU6(inplace=True),
        )
    )

def conv1x1(ch_in, ch_out):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )

def conv3x3(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )




# Resnet Block
class ResnetBlock(nn.Module):
    
    def __init__(self, input_channels, out_channels=None):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.instancenorm = nn.InstanceNorm2d(input_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        
        original_x = x.clone()
        x = self.conv1(x)
        x = self.instancenorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instancenorm(x)
        return original_x + x



# Inverted Block
class InvertedBlock(nn.Module):
    def __init__(self, input_channels, out_channels=None, expand_ratio = 4, stride=1):
        super(InvertedBlock, self).__init__()
        
        self.out_channels = out_channels
        hidden_dim = 64 * expand_ratio

        self.DWconv = nn.Sequential(
            conv1x1(input_channels, hidden_dim),
            dwise_conv(hidden_dim, stride=stride),
            conv1x1(hidden_dim, input_channels)
        )


        if self.out_channels != None:
            self.DWconv2 = nn.Sequential(
                conv1x1(input_channels, hidden_dim),
                dwise_conv(hidden_dim, stride=stride),
                conv1x1(hidden_dim, out_channels)
            )


    def forward(self, x):    
        if self.out_channels != None:
            return self.DWconv2(x)
        original_x = x.clone()
        x = self.DWconv(x)
        return original_x + x


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, input_channels, out_channels=None, expand_ratio = 3, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.out_channels = out_channels
        hidden_dim = input_channels // expand_ratio

        self.DWconv = nn.Sequential(
            conv1x1(input_channels, hidden_dim),
            dwise_conv(hidden_dim, stride=stride),
            conv1x1(hidden_dim, input_channels)
        )


        if self.out_channels != None:
            self.DWconv2 = nn.Sequential(
                conv1x1(input_channels, hidden_dim),
                dwise_conv(hidden_dim, stride=stride),
                conv1x1(hidden_dim, out_channels)
            )


    def forward(self, x):    
        if self.out_channels != None:
            return self.DWconv2(x)
        original_x = x.clone()
        x = self.DWconv(x)
        return original_x + x



##### Contracting and Expanding Blocks


class ContractingBlock(nn.Module):
    
    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation='relu'):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, padding=1, stride=2, padding_mode='reflect')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2)
        self.use_bn = use_bn

    def forward(self, x):

        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class ExpandingBlock(nn.Module):
 
    def __init__(self, input_channels, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()

    def forward(self, x):
  
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=3, padding_mode='reflect')

    def forward(self, x):
    
        x = self.conv(x)
        return x