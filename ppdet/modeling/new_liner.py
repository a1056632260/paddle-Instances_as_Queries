import paddle
from ...fluid.dygraph import Flatten  # noqa: F401
from paddle.nn import functional as F

from paddle.nn import Layer






class Linear2(Layer):
    r"""

    Fully-connected linear transformation layer. For each input :math:`X` ,
    the equation is:

    .. math::

        Out = XW + b

    where :math:`W` is the weight and :math:`b` is the bias.

    Linear layer takes only one multi-dimensional tensor as input with the
    shape :math:`[batch\_size, *, in\_features]` , where :math:`*` means any
    number of additional dimensions. It multiplies input tensor with the weight
    (a 2-D tensor of shape :math:`[in\_features, out\_features]` ) and produces
    an output tensor of shape :math:`[batch\_size, *, out\_features]` .
    If :math:`bias\_attr` is not False, the bias (a 1-D tensor of
    shape :math:`[out\_features]` ) will be created and added to the output.

    Parameters:
        in_features (int): The number of input units.
        out_features (int): The number of output units.
        weight_attr (ParamAttr, optional): The attribute for the learnable
            weight of this layer. The default value is None and the weight will be
            initialized to zero. For detailed information, please refer to
            paddle.ParamAttr.
        bias_attr (ParamAttr|bool, optional): The attribute for the learnable bias
            of this layer. If it is set to False, no bias will be added to the output.
            If it is set to None or one kind of ParamAttr, a bias parameter will
            be created according to ParamAttr. For detailed information, please refer
            to paddle.ParamAttr. The default value is None and the bias will be
            initialized to zero.
        name (str, optional): Normally there is no need for user to set this parameter.
            For detailed information, please refer to :ref:`api_guide_Name` .

    Attribute:
        **weight** (Parameter): the learnable weight of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Shape:
        - input: Multi-dimentional tensor with shape :math:`[batch\_size, *, in\_features]` .
        - output: Multi-dimentional tensor with shape :math:`[batch\_size, *, out\_features]` .

    Examples:
        .. code-block:: python

          import paddle

          # Define the linear layer.
          weight_attr = paddle.ParamAttr(
              name="weight",
              initializer=paddle.nn.initializer.Constant(value=0.5))
          bias_attr = paddle.ParamAttr(
              name="bias",
              initializer=paddle.nn.initializer.Constant(value=1.0))
          linear = paddle.nn.Linear(2, 4, weight_attr=weight_attr, bias_attr=bias_attr)
          # linear.weight: [[0.5 0.5 0.5 0.5]
          #                 [0.5 0.5 0.5 0.5]]
          # linear.bias: [1. 1. 1. 1.]

          x = paddle.randn((3, 2), dtype="float32")
          # x: [[-0.32342386 -1.200079  ]
          #     [ 0.7979031  -0.90978354]
          #     [ 0.40597573  1.8095392 ]]
          y = linear(x)
          # y: [[0.23824859 0.23824859 0.23824859 0.23824859]
          #     [0.9440598  0.9440598  0.9440598  0.9440598 ]
          #     [2.1077576  2.1077576  2.1077576  2.1077576 ]]
    """

    def __init__(self,
                 in_features,
                 out_features,
                 weight_attr=None,
                 bias_attr=None,
                 name=None):
        super(Linear2, self).__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.weight = self.create_parameter(
            shape=[out_features, in_features],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.bias = self.create_parameter(
            shape=[in_features],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True)
        self.name = name

    def forward(self, input):
        out = F.linear(
            x=self.weight, weight=input, bias=self.bias, name=self.name)
        return out

    def extra_repr(self):
        name_str = ', name={}'.format(self.name) if self.name else ''
        return 'in_features={}, out_features={}, dtype={}{}'.format(
            self.weight.shape[0], self.weight.shape[1], self._dtype, name_str)

