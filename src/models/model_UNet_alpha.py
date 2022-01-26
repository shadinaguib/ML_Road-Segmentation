"""
Modules for U-Net Alpha model

Same structure as the original UNet, except we add a zero padding to retain the input dimension in the output prediction
"""

from helpers.config import *  # Import PADDING_MODE and channels sizes 
import torch
from torch.nn import Module, ModuleList, Sequential, ConvTranspose2d, Conv2d, MaxPool2d, BatchNorm2d, ReLU, Sigmoid


"""
NUM_CLASSES=1
PADDING_MODE = 'replicate'

ENC_CHANNELS = (3, 16, 32, 64, 128, 256)
DEC_CHANNELS = (256, 128, 64, 32, 16)
"""

class ConvBlock(Module): 
  """
  Building block for encoder and decoder part of the UNet models
  Two 3x3 convolution with zero padding, Batchnorm and ReLU activated
  """
  def __init__(self, in_c, out_c):
    super().__init__() 
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
    return self.conv2(self.conv1(x))



class Encoder(Module):
  """
  Class for the Encoder part of the model
  Encoder channels = (3, 16, 32, 64, 128, 256), 5 levels of depth

  Apply the ConvBlock 5 times with max pooling between each
  MaxPooling divide image size by 2
  """
  def __init__(self, channels): # channels = (3, 16, 32, 64, 128, 256)
    super().__init__()
    # Create the list of encoding modules
    self.enc_blocks = ModuleList([ConvBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
    self.pool = MaxPool2d(2)

  def forward(self, x):
    """
    Take RGB image as input
    Returns the result of the encoding steps, with the list of intermediate feature maps
    """
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


class UNetAlpha(Module):
  """
  Class for the whole UNet Alpha model
  enc_c = (3, 16, 32, 64, 128, 256)
  dec_c = (256, 128, 64, 32, 16)
  nb_classes = 1

  Apply the encoding and decoding steps, and classiy the output with Sigmoid
  """
  def __init__(self, enc_c=ENC_CHANNELS, dec_c=DEC_CHANNELS, nb_classes=NUM_CLASSES):
    super().__init__()
    # Initialize the encoder and decoder
    self.encoder = Encoder(enc_c)
    self.decoder = Decoder(dec_c) 
    # Initialize the classifier head, last step to generate the segmentation map
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
