"""
Modules for U-Net Gamma model

Same structure as the original UNet, except we add:
- zero padding to retain input dimensions
- dilated convolution to the first two encoding blocks of U-Net Alpha
- ResPath with gradually decreasing number of convolution along skip connections
"""

from helpers.config import *  # Import PADDING_MODE, channels sizes and respath lengths
import torch
from torch.nn import Module, ModuleList, Sequential, ConvTranspose2d, Conv2d, MaxPool2d, BatchNorm2d, ReLU, Sigmoid


class ConvBlock(Module): 
  """
  Building block for encoder and decoder part of the UNet models
  Two 3x3 convolution with zero padding, Batchnorm and ReLU activated
  The first convolution is dilated for the first two encoding blocks:
    - dilation rate = 3 for the first
    - dilation rate = 2 for the second
  """
  def __init__(self, in_c, out_c, step=2):  
    super().__init__()
    if step == 0: # conv77 like = conv33 with dilatation=2
        self.conv1 = Sequential(Conv2d(in_c, out_c, kernel_size=3, padding=3, padding_mode=PADDING_MODE, dilation=3),
                                BatchNorm2d(out_c),
                                ReLU())
    elif step == 1: # conv55 like = conv33 with dilatation=1
        self.conv1 = Sequential(Conv2d(in_c, out_c, kernel_size=3, padding=2, padding_mode=PADDING_MODE, dilation=2),
                                BatchNorm2d(out_c),
                                ReLU())
    elif step >= 2: # conv33 after second step
        self.conv1 = Sequential(Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode=PADDING_MODE),
                                BatchNorm2d(out_c),
                                ReLU())

    self.conv2 = Sequential(Conv2d(out_c, out_c, kernel_size=3, padding=1, padding_mode=PADDING_MODE),
                            BatchNorm2d(out_c),
                            ReLU())
    
  def forward(self, x):
    """
    Take an image in input
    Returns the image convoluted two times, with same dimension:
    CONV => BatchNorm => ReLU => CONV => BatchNorm => ReLU 
    """
    return(self.conv2(self.conv1(x)))



class ResPath(Module):
  """
  Residual path for skip connections between encoder and decoder
  
  Apply a series of convolutional blocks to the saved features map of the encoder
  The number of convolutions depend on the depth of the path:
   - First ResPath  (16 features):  4 blocks of 3x3 convolutions with 1x1 residual
   - Second ResPath (32 features):  3 blocks of 3x3 convolutions with 1x1 residual
   - Third ResPath  (64 features):  2 blocks of 3x3 convolutions with 1x1 residual
   - Fourth ResPath (128 features): 1 blocks of 3x3 convolutions with 1x1 residual

  channel = number of filters at this step
  respath_length = number of convolution blocks for the current respath
  """
  def __init__(self, channel, respath_length): 
      super().__init__()
      self.respath_length = respath_length
      self.shortcut = Sequential(Conv2d(channel, channel, kernel_size=1, padding=0),
                                BatchNorm2d(channel))
      self.conv = Sequential(Conv2d(channel, channel, kernel_size=3, padding=1, padding_mode=PADDING_MODE),
                            BatchNorm2d(channel),
                            ReLU())
      self.bn_relu = Sequential(BatchNorm2d(channel, affine=False),
                                ReLU())

  def forward(self, x):
    """
    Take the feature map output of an encoding block as input
    Returns the output of the ResPath
    """
    for i in range(self.respath_length):
        # Residual connection for each convolution
        shortcut = self.shortcut(x)
        x = self.conv(x)
        # Include of the residual connection to the output of convolution
        x = x + shortcut
        # Apply Batchnorm and ReLU to the new output
        x = self.bn_relu(x)
    return x


class Encoder(Module):
  """
  Class for the Encoder part of the model
  Encoder channels = (3, 16, 32, 64, 128, 256), 5 levels of depth
  ResPath Lengths = (4, 3, 2, 1)

  Apply the ConvBlock 5 times with max pooling between each
  MaxPooling divide image size by 2
  """
  def __init__(self, channels, respath_lengths): # channels = (3, 16, 32, 64, 128, 256), respath_lengths=(4, 3, 2, 1)
    super().__init__()
    # Create the list of encoding modules and respath modules
    self.enc_blocks = ModuleList([ConvBlock(channels[i], channels[i + 1], step=i) for i in range(len(channels) - 1)])
    self.res_paths = ModuleList([ResPath(channels[i], respath_lengths[i-1]) for i in range(1,len(channels)-1)])
    self.pool = MaxPool2d(2)

  def forward(self, x):
    # Empty list to store the intermediate outputs for skip connections
    block_outputs = []
    down_step = len(self.enc_blocks) 
    # Loop through the encoder blocks
    for block in self.enc_blocks:
      down_step -=  1
      # Pass the inputs through the current encoder block, store the outputs, and then apply maxpooling on the output
      x = block(x)
      # If bottom of UNet, no need for MaxPooling
      if down_step == 0:
        break

      block_outputs.append(x)
      x = self.pool(x) 
    # Return the last output x and the list containing the intermediate outputs
    return x, block_outputs


class Decoder(Module):
  def __init__(self, channels):
    super().__init__()
  """
  Class for the Encoder part of the model
  Decoder channels = (256, 128, 64, 32, 16), 5 levels of depth

  Apply the Transposed Convolution 4 times, with ConvBlocks after each
  Transposed Convolution multiply image size by 2
  """
  def __init__(self, channels): # channels = (256, 128, 64, 32, 16)
    super().__init__()
    # Create the list of decoding modules: upsample operations and convolutions
    self.num_c = len(channels)
    self.up_blocks = ModuleList([ConvTranspose2d(channels[i], channels[i + 1], kernel_size=2, stride=2, padding=0) for i in range(self.num_c-1)])
    self.dec_blocks = ModuleList([ConvBlock(channels[i], channels[i + 1]) for i in range(self.num_c-1)])

  def forward(self, x, enc_features):
    """
    Take image from the last encoding block and saved intermediate feature maps as input
    Returns output of the model, just before the classifier convolution
    """
    # Loop through the list of channels
    for i in range(self.num_c - 1):
      # Pass the inputs through the upsampler block
      x = self.up_blocks[i](x)
      # Concatenate saved feature map from corresponding encoder step with current decoder feature map
      x = torch.cat([x, enc_features[len(enc_features)-i-1]], dim=1) 
      # Pass through the decoder convolution block
      x = self.dec_blocks[i](x)
    return x


class UNetGamma(Module):
  """
  Class for the whole UNet beta model
  enc_c = (3, 16, 32, 64, 128, 256)
  dec_c = (256, 128, 64, 32, 16)
  respath_lengths = (4, 3, 2, 1)
  nb_classes = 1

  Apply the encoding and decoding steps, and classiy the output with Sigmoid
  """
  def __init__(self, enc_c=ENC_CHANNELS, dec_c=DEC_CHANNELS, nb_classes=NUM_CLASSES, respath_lengths=RESPATHS_LENGTHS):
    super().__init__()
    # Initialize the encoder and decoder
    self.encoder = Encoder(enc_c, respath_lengths)
    self.decoder = Decoder(dec_c)
    # Initialize the regression head, last step to generate the segmentation map
    self.head = Sequential(Conv2d(in_channels=dec_c[-1], out_channels=nb_classes, kernel_size=1),
                Sigmoid())

  def forward(self, x):
    """
    Take input RGB image as input
    Returns the predicted ground-truth with same dimensions as input image
    """
    x, enc_features = self.encoder(x)
    # Pass the encoder features through decoder with dimensions suited for concatenation
    x = self.decoder(x, enc_features)
    # Pass the decoder features through the regression head to obtain the segmentation map
    gt = self.head(x)
    return gt
