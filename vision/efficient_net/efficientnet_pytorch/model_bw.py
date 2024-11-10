"""model.py - Model and module class for EfficientNet with Batch Whitening.
   They are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).
# with Batch Whitening by guyk1971 (github username)


import torch
from torch import nn
from torch.nn import functional as F
from .utils_bw import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)



VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',

    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2'
)

from .batch_whitening import *



class BWCholeskyBlock(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features,momentum=0.1,eps=1e-5,pre_bias_block=None,num_bias_features=None):
        super().__init__()
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively 
        self.n_features=num_features
        self.n_bias_features = num_features if pre_bias_block is None else num_bias_features
        self.momentum = momentum
        self.eps = eps
        self.cov_warmup=False
        # self.gamma = nn.Parameter(torch.ones(num_features))
        # The variables that are not model parameters are initialized to 0 and 1
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_cov', torch.eye(num_features))
        self.pre_bias_block=pre_bias_block

        self.beta = nn.Parameter(torch.zeros(self.n_bias_features))

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.running_mean.device != X.device:
            self.running_mean = self.running_mean.to(X.device)
            self.running_cov = self.running_cov.to(X.device)
        # Save the updated running_mean and moving_var
        Y, self.running_mean, self.running_cov = cholesky_batch(
            X, self.running_mean, self.running_cov, eps=self.eps, momentum=self.momentum,cov_warmup=self.cov_warmup)
        if self.pre_bias_block is not None:
            Y=self.pre_bias_block(Y)
        # add the bias
        shape = (1,self.n_bias_features) if len(X.shape)==2 else (1,self.n_bias_features,1,1)
        Y += self.beta.view(shape)
        return Y


class BWItnBlock(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features,T=10,momentum=0.1,eps=1e-5,pre_bias_block=None,num_bias_features=None):
        super().__init__()
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively 
        self.T=T
        self.n_features=num_features
        self.n_bias_features = num_features if pre_bias_block is None else num_bias_features
        self.momentum = momentum
        self.eps = eps

        # note: currently assuming no grouping of features so we have only one group
        num_groups=1
        self.num_channels = num_features
        # The variables that are not model parameters are initialized to 0 and 1
        self.register_buffer('running_mean', torch.zeros(num_groups, self.num_channels, 1))
        self.register_buffer('running_cov', torch.eye(self.num_channels).expand(num_groups, self.num_channels, self.num_channels))
        self.pre_bias_block=pre_bias_block

        # self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(self.n_bias_features))

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.running_mean.device != X.device:
            self.running_mean = self.running_mean.to(X.device)
            self.running_cov = self.running_cov.to(X.device)
        # Save the updated running_mean and moving_var
        X_hat,self.running_mean,self.running_cov = iter_norm_batch(X,
                                                                  self.running_mean,
                                                                  self.running_cov,
                                                                  self.T,
                                                                  momentum=self.momentum,
                                                                  n_channels=self.num_channels)
        if self.pre_bias_block is not None:
            X_hat=self.pre_bias_block(X_hat)
        # add the bias
        shape = (1,self.n_bias_features) if len(X.shape)==2 else (1,self.n_bias_features,1,1)
        X_hat += self.beta.view(shape)
        return X_hat


# BatchWhiteningBlock=BWItnBlock
BatchWhiteningBlock=BWCholeskyBlock



class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        Conv2d = get_same_padding_conv2d(image_size=(1, 1))
        self._se_reduce = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self._se_expand = Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1)
        self._swish = MemoryEfficientSwish()
    
    def forward(self, inputs):
        x=inputs
        x_squeezed = F.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self._swish(x_squeezed)
        x_squeezed = self._se_expand(x_squeezed)
        x = torch.sigmoid(x_squeezed) * x
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()





class MBConvBlockBW(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        block_type = global_params.mbconv_type
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self._bw_mom = 1 - global_params.batch_whitening_momentum  # pytorch's difference from tensorflow
        self._bw_eps = global_params.batch_whitening_epsilon


        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            if block_type==0:
                self._expand_conv = nn.Sequential(
                    Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False),
                    nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
                )
            else:
                pre_bias_block = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
                self._expand_conv = BatchWhiteningBlock(num_features=inp, momentum=self._bw_mom, eps=self._bw_eps,pre_bias_block=pre_bias_block,num_bias_features=oup)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        if block_type==1:
            self._depthwise_block=nn.Sequential(
                nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps),
                Conv2d(in_channels=oup, out_channels=oup, groups=oup,kernel_size=k, stride=s, bias=False)  # depthwise conv
            )
        elif block_type==2 or block_type==0:
            self._depthwise_block=nn.Sequential(
                Conv2d(in_channels=oup, out_channels=oup, groups=oup,kernel_size=k, stride=s, bias=False),  # depthwise conv
                nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps),
            )
        elif block_type==3:
            self._depthwise_block=Conv2d(in_channels=oup, out_channels=oup, groups=oup,kernel_size=k, stride=s, bias=False)  # depthwise conv
     
        image_size = calculate_output_image_size(image_size, s)
        self._swish = MemoryEfficientSwish()
        # Squeeze and Excitation layer, if desired
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)     # Pointwise convolution phase

        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_block = SEBlock(in_channels=oup,out_channels=num_squeezed_channels,kernel_size=1)
            pre_bias_block = nn.Sequential(
                self._se_block,
                self._project_conv
            )
        else:
            pre_bias_block = nn.Sequential(
                self._project_conv
            )

        if block_type==0:
            self._proj_block = nn.Sequential(
                pre_bias_block,
                nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
            )
        else:
            self._proj_block = BatchWhiteningBlock(num_features=oup, momentum=self._bw_mom, eps=self._bw_eps,pre_bias_block=pre_bias_block,num_bias_features=final_oup)
        

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._swish(x)


        x = self._depthwise_block(x)
        x = self._swish(x)


        x = self._proj_block(x)
        # x = self._swish(x)          # is it necessary ? didnt appear in original implementation


        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        if self.has_se:
            self._se_block.set_swish(memory_efficient=memory_efficient)




class EfficientNetBW(nn.Module):
    """EfficientNet model using Batch Whitening.
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    Example:
        >>> import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Batch whitening parameters
        bw_mom = 1 - self._global_params.batch_whitening_momentum
        bw_eps = self._global_params.batch_whitening_epsilon


        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        
        if self._global_params.mbconv_type==0:       # mbconv_type=0 means without BW. run the original model as is
            self._conv_stem_block = nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
            )
        elif self._global_params.conv_stem_type==1:
            self._conv_stem_block=nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
            )
        elif self._global_params.conv_stem_type==2:
            pre_bias_block = Conv2d(in_channels,out_channels,kernel_size=1,stride=2)
            self._conv_stem_block=nn.Sequential(
                Conv2d(in_channels, in_channels, kernel_size=3, stride=2, bias=False,groups=in_channels),
                BatchWhiteningBlock(num_features=in_channels, momentum=bw_mom, eps=bw_eps,pre_bias_block=pre_bias_block,num_bias_features=out_channels)
            )

        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlockBW(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlockBW(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        pre_bias_block = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        if self._global_params.mbconv_type==0:       # mbconv_type=0 means without BW. run the original model as is
            self._conv_head_block=nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
            )
        else:
            self._conv_head_block=BatchWhiteningBlock(num_features=in_channels, momentum=bw_mom, eps=bw_eps,pre_bias_block=pre_bias_block,num_bias_features=out_channels)


        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        # set activation to memory efficient swish by default
        self._swish = MemoryEfficientSwish()

        # create a list of all BW layers in the model
        self.bw_layers = self._get_bw_layers()
        self.curr_cov_warmup=False
        self.set_bw_cov_warmup(True)

    def _get_bw_layers(self):
        bw_layers = []

        def _extract_layers_recursive(module):
            for name, submodule in module.named_children():

                if isinstance(submodule, BatchWhiteningBlock):
                    bw_layers.append(submodule)
                # If the submodule has children, recursively call this function
                if len(list(submodule.children())) > 0:
                    _extract_layers_recursive(submodule)

        _extract_layers_recursive(self)
        return bw_layers

    def set_bw_cov_warmup(self,cov_warmup):
        if self.curr_cov_warmup!=cov_warmup:
            for layer in self.bw_layers:
                    layer.cov_warmup = cov_warmup        
            self.curr_cov_warmup = cov_warmup
        return
    

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])
                >>> print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = inputs
        # x = self._bw0(x)
        x = self._conv_stem_block(x)
        # x = self._bn0(x)
        x = self._swish(x)
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x

        # Head
        # x = self._bw1(x)
        x = self._conv_head_block(x)
        # x = self._bn1(x)
        x = self._swish(x)

        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        return endpoints

    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = inputs
        # x = self._bw0(x)
        x = self._conv_stem_block(x)
        # x = self._bn0(x)
        x = self._swish(x)

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        # x = self._bw1(x)
        x = self._conv_head_block(x)
        # x = self._bn1(x)
        x = self._swish(x)

        return x

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        x = self.extract_features(inputs)
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """Create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, **override_params):
        """Create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            advprop (bool):
                Whether to load pretrained weights
                trained with advprop (valid when weights_path is None).
            in_channels (int): Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            A pretrained efficientnet model.
        """
        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path,
                                load_fc=(num_classes == 1000), advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given efficientnet model.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            bool: Is a valid name or not.
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.

        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
