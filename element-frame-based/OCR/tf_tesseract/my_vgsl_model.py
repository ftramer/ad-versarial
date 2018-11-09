import re
import tensorflow as tf
import tensorflow.contrib.slim as slim
from OCR.tf_tesseract import shapes
from OCR.tf_tesseract import vgslspecs


MODEL_SPEC = '1,0,0,1[Ct3,3,16 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx512]O1c111'
COLOR_MODEL_SPEC = '1,0,0,3[Ct3,3,16 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx512]O1c111'


def _ParseOutputSpec(output_spec):
    """Parses the output spec.

    Args:
      output_spec: Output layer definition. See Build.

    Returns:
      out_dims:     2|1|0 for 2-d, 1-d, 0-d.
      out_func:     l|s|c for logistic, softmax, softmax+CTC
      num_classes:  Number of classes in output.

    Raises:
      ValueError: if syntax is incorrect.
    """
    pattern = re.compile(R'(O)(0|1|2)(l|s|c)(\d+)')
    m = pattern.match(output_spec)
    if m is None:
        raise ValueError('Failed to parse output spec:' + output_spec)
    out_dims = int(m.group(2))
    out_func = m.group(3)
    if out_func == 'c' and out_dims != 1:
        raise ValueError('CTC can only be used with a 1-D sequence!')
    num_classes = int(m.group(4))
    return out_dims, out_func, num_classes


def parse_specs(model_spec):
    left_bracket = model_spec.find('[')
    right_bracket = model_spec.rfind(']')
    if left_bracket < 0 or right_bracket < 0:
        raise ValueError('Failed to find [] in model spec! ', model_spec)
    layer_spec = model_spec[left_bracket:right_bracket + 1]
    output_spec = model_spec[right_bracket + 1:]
    return layer_spec, output_spec


class MyVGSLImageModel(object):
    def __init__(self, color=False, use_gpu=True):
        self.model_spec = COLOR_MODEL_SPEC if color else MODEL_SPEC
        self.input = None
        # The layers between input and output.
        self.layers = None
        # Tensor for loss
        self.loss = None
        # Train operation
        self.train_op = None
        # Tensor for the output predictions (usually softmax)
        self.output = None
        self.logits = None
        # True if we are using CTC training mode.
        self.using_ctc = False
        self.layer_spec, self.output_spec = parse_specs(self.model_spec)
        self.ctc_width = None
        self.use_gpu = use_gpu

    def __call__(self, images, heights, widths, reuse=None):
        self.Build(images, heights, widths, reuse)
        return self.logits, self.output

    def Build(self, images, heights, widths, reuse=None):
        out_dims, out_func, num_classes = _ParseOutputSpec(self.output_spec)
        self.using_ctc = out_func == 'c'
        self.input = images
        self.layers = vgslspecs.VGSLSpecs(widths, heights, False, self.use_gpu)
        with tf.variable_scope('vgsl_model', values=[self.input]) as sc:
            last_layer = self.layers.Build(images, self.layer_spec, reuse)
            self._AddOutputs(last_layer, out_dims, out_func, num_classes, reuse, sc)

    def _AddOutputs(self, prev_layer, out_dims, out_func, num_classes, reuse=None, sc=None):
        logits, outputs = self._AddOutputLayer(prev_layer, out_dims, out_func,
                                               num_classes, reuse=reuse, sc=sc)
        height_in = shapes.tensor_dim(prev_layer, dim=1)
        self.ctc_width = self.layers.GetLengths(dim=2, factor=height_in)
        self.logits = logits
        self.output = outputs

    def _AddOutputLayer(self, prev_layer, out_dims, out_func, num_classes, reuse=None, sc=None):
        # Reduce dimensionality appropriate to the output dimensions.
        batch_in = shapes.tensor_dim(prev_layer, dim=0)
        height_in = shapes.tensor_dim(prev_layer, dim=1)
        width_in = shapes.tensor_dim(prev_layer, dim=2)
        depth_in = shapes.tensor_dim(prev_layer, dim=3)
        if out_dims:
            # Combine any remaining height and width with batch and unpack after.
            shaped = tf.reshape(prev_layer, [-1, depth_in])
        else:
            # Everything except batch goes to depth, and therefore has to be known.
            shaped = tf.reshape(prev_layer, [-1, height_in * width_in * depth_in])
        logits = slim.fully_connected(shaped, num_classes, activation_fn=None, reuse=reuse, scope=sc)
        if out_func == 'l':
            raise ValueError('Logistic not yet supported!')
        else:
            output = tf.nn.softmax(logits)
        # Reshape to the dessired output.
        if out_dims == 2:
            output_shape = [batch_in, height_in, width_in, num_classes]
        elif out_dims == 1:
            output_shape = [batch_in, height_in * width_in, num_classes]
        else:
            output_shape = [batch_in, num_classes]

        output = tf.reshape(output, output_shape, name='Output')
        logits = tf.reshape(logits, output_shape, name='Logits')
        return logits, output


def ctc_loss(logits, widths, sparse_labels, use_baidu=False):
    ctc_input = tf.transpose(logits, [1, 0, 2])
    if use_baidu:
        with tf.get_default_graph()._kernel_label_map({"CTCLoss": "WarpCTC"}):
            cross_entropy = tf.nn.ctc_loss(inputs=ctc_input, labels=sparse_labels,
                                           sequence_length=widths)
    else:
        cross_entropy = tf.nn.ctc_loss(inputs=ctc_input, labels=sparse_labels,
                                       sequence_length=widths)
    return tf.reduce_sum(cross_entropy, name='loss')

def ctc_decode(logits, widths, beam=False):
    ctc_input = tf.transpose(logits, [1, 0, 2])
    if beam:
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(ctc_input, widths)
    else:
        decoded, log_prob = tf.nn.ctc_greedy_decoder(ctc_input, widths)
    return decoded
