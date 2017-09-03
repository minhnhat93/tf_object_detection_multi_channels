# Tutorial on How to change tensorflow object detection API to allow any number of input channels

This tutorial was written for the following commit of tensorflow object detection API:
https://github.com/tensorflow/models/tree/2243d30cd9be6a57f8f23d398d702a3ff7aacf11/object_detection

For many reasons, I cannot publish the modified object detection API that I am using.
Therefore, I am writing this tutorial. This tutorial is mainly for FasterRCNNResnet101.
RFCN use the same feature extractor as FasterRCNN so it should be almost the same.
Other architectures (SSD) should be similar. If I am missing something, please send me a message.

## Preparing the input:
### Create tfrecord:
You need to change the format of your .tfrecord file to store a multi-dimensional
array instead of an image like in the example at https://github.com/tensorflow/models/blob/2243d30cd9be6a57f8f23d398d702a3ff7aacf11/object_detection/create_pascal_tf_record.py

Instead of these two line in the file above:

    with tf.gfile.GFile(full_path, 'rb') as fid:
      encoded_jpg = fid.read()

You need to prepare your inputs as a numpy array and encode it as string like this example:

    # Read your image and extra inputs
    image = cv2.imread('path/to/image')[:, :, ::-1]
    background = cv2.imread('path/to/background')[:, :, ::-1]
    # Image and background are numpy arrays that has dimension of H x W x 3
    # Concatenate them on depth channel to create an H x W x 6 input
    inputs_stacked = np.concatenate([image, background], axis=-1)
    # Encode your input as string
    encoded_inputs = inputs_stacked.tostring()

Next, when you create tf.train.Example object, you must store your new input as bytes_feature.
Also, you need to store the number of channels in your input too. The way to
do this is the same when you store image width and height.

    tf_example = tf.train.Example(features=tf.train.Features(feature={
    ...
        'image/channels': tf.FixedLenFeature((), tf.int64, 1),
        'image/encoded': dataset_util.bytes_feature(encoded_inputs),
    ...
      }))

### Change the data decoder:
You must change the file object_detection/data_decoders/tf_example_decoder.py
to allow it to read your new input:

First, in the class TFExampleDecoder, add this function to read your input:

    def _read_image(self, keys_to_tensors):
      image_encoded = keys_to_tensors['image/encoded']
      height = keys_to_tensors['image/height']
      width = keys_to_tensors['image/width']
      channels = keys_to_tensors['image/channels']
      to_shape = tf.cast(tf.stack([height, width, channels]), tf.int32)
      image = tf.reshape(tf.decode_raw(image_encoded, tf.uint8), to_shape)
      return image

This function reshapes your encoded input to a 3D array using the number of channels information
that you stored earlier

Now, in the __init__ function of the same class, change the item_to_handler of your encoded image at this line:
https://github.com/tensorflow/models/blob/2243d30cd9be6a57f8f23d398d702a3ff7aacf11/object_detection/data_decoders/tf_example_decoder.py#L56

to this:

        fields.InputDataFields.image:
            slim_example_decoder.ItemHandlerCallback(
              keys=['image/encoded', 'image/height', 'image/width', 'image/channels'],
              func=self._read_image
            )

## Change architecture code:

You need to change a little bit of architecture code to make the API know that the input
for tfrecord file have more than 3 input channels:

### Change protobuf file:
In the file object_detection/protos/faster_rcnn.proto, add this line to the end of
message FasterRcnn:

    optional uint32 num_input_channels = 28 [default=3];

Now, compile your protobuf files again with:

    protoc object_detection/protos/*.proto --python_out=.

### Change signature of the feature extractor meta architecture:

You need to open the file that contains the meta architecture of your model in
the directory object_detection/meta_architectures and change the signature of the
FeatureExtractore class by adding num_input_channel so that your feature extractor
know how many channel your input have.

For example, if you use FasterRCNNResnet101, open the file
object_detection/meta_architectures/faster_rcnn_meta_arch.py

Next, look for the __init__ function of your feature extractor and add num_input_channels
to its signature. Also, store the channels as an attribute of the object.
For FasterRCNN, you would need to look at line 88 (https://github.com/tensorflow/models/blob/2243d30cd9be6a57f8f23d398d702a3ff7aacf11/object_detection/meta_architectures/faster_rcnn_meta_arch.py#L88)
 and change the init function to:

      def __init__(self,
                   is_training,
                   first_stage_features_stride,
                   num_input_channels=3,
                   reuse_weights=None,
                   weight_decay=0.0):
        """Constructor.

        Args:
          is_training: A boolean indicating whether the training version of the
            computation graph should be constructed.
          first_stage_features_stride: Output stride of extracted RPN feature map.
          reuse_weights: Whether to reuse variables. Default is None.
          weight_decay: float weight decay for feature extractor (default: 0.0).
        """
        self._is_training = is_training
        self._first_stage_features_stride = first_stage_features_stride
        self._num_input_channels = num_input_channels
        self._reuse_weights = reuse_weights
        self._weight_decay = weight_decay

### Change feature extractor code:

You need to change your feature extractor code in the directory
object_detection/models/. For FasterRCNNResnet101, you need to open
the file faster_rcnn_resnet_v1_feature_extractor.py.

Look for the class of your feature extractor. For example, I am using ResnetV1,
so I will look for the class class FasterRCNNResnetV1FeatureExtractor at line 40
(https://github.com/tensorflow/models/blob/2243d30cd9be6a57f8f23d398d702a3ff7aacf11/object_detection/models/faster_rcnn_resnet_v1_feature_extractor.py#L40)
Again, add num_input_channels to the **init** function signature. Next, add num_init_channel to
the **super** call too:

      def __init__(self,
                   architecture,
                   resnet_model,
                   is_training,
                   first_stage_features_stride,
                   num_input_channels=3,
                   reuse_weights=None,
                   weight_decay=0.0):
        """Constructor.

        Args:
          architecture: Architecture name of the Resnet V1 model.
          resnet_model: Definition of the Resnet V1 model.
          is_training: See base class.
          first_stage_features_stride: See base class.
          reuse_weights: See base class.
          weight_decay: See base class.

        Raises:
          ValueError: If `first_stage_features_stride` is not 8 or 16.
        """
        if first_stage_features_stride != 8 and first_stage_features_stride != 16:
          raise ValueError('`first_stage_features_stride` must be 8 or 16.')
        self._architecture = architecture
        self._resnet_model = resnet_model
        super(FasterRCNNResnetV1FeatureExtractor, self).__init__(
            is_training, first_stage_features_stride, num_input_channels, reuse_weights, weight_decay)

You also need to change the preprocess function in the same class (line 67)
because this function subtracts the mean for each channel. Basically, you may
would want to add zeros at the end for new channels or just reuse the value
of the RGB channels. Choose the mean values carefully for ResnetV1 because
it affect the performance a lot.

There's one more class init that you need to change in the same file. It's a
subclass of the class that you just changed above. The name of this class depend
on which model you choose. For FasterRCNNResnet101, the class name will be
FasterRCNNResnet101FeatureExtractor. Just like above, add num_input_channels to the signature
and the super call:

      def __init__(self,
                   is_training,
                   first_stage_features_stride,
                   num_input_channels=3,
                   reuse_weights=None,
                   weight_decay=0.0):
        """Constructor.

        Args:
          is_training: See base class.
          first_stage_features_stride: See base class.
          reuse_weights: See base class.
          weight_decay: See base class.

        Raises:
          ValueError: If `first_stage_features_stride` is not 8 or 16,
            or if `architecture` is not supported.
        """
        super(FasterRCNNResnet101FeatureExtractor, self).__init__(
            'resnet_v1_101', resnet_v1.resnet_v1_101, is_training,
            first_stage_features_stride, num_input_channels, reuse_weights, weight_decay)

### Change model_builder.py:

That's almost it for the architecture code. You also need to change the file
object_detection/builders/model_builder.py.
How you change this file depend on which architecture you use.
If you use SSD you will need to change different function from this one
but the spirit is the same.
Change function signature of _build_faster_rcnn_feature_extractor at this line:
https://github.com/tensorflow/models/blob/2243d30cd9be6a57f8f23d398d702a3ff7aacf11/object_detection/builders/model_builder.py#L164

to (basically just add *num_input_channels=3*)

    def _build_faster_rcnn_feature_extractor(feature_extractor_config, is_training, num_input_channels=3, reuse_weights=None):

Next, change the return line of the same function at https://github.com/tensorflow/models/blob/2243d30cd9be6a57f8f23d398d702a3ff7aacf11/object_detection/builders/model_builder.py#L189

to

    return feature_extractor_class(is_training, first_stage_features_stride, num_input_channels, reuse_weights)

Next, look for function _build_faster_rcnn_model, add this line to its body:

    num_input_channels = frcnn_config.num_input_channels

Chage the feature extractor build call in the same function at line 213
(https://github.com/tensorflow/models/blob/2243d30cd9be6a57f8f23d398d702a3ff7aacf11/object_detection/builders/model_builder.py#L213)
to this:

    feature_extractor = _build_faster_rcnn_feature_extractor(
        frcnn_config.feature_extractor, is_training, num_input_channels)

That's it.

## Config file:

In the config file, make sure to add the num_input_channels under the faster_rcnn message.

## Export inference graph:

In the exporter.py file, look for function _image_tensor_input_placeholder().
Change the shape of the returned place holder to (None, None, None, num_input_channels).
For tf_example input, you don't need to specify this shape.

## Transfer learning:

Because the shape of the first convolution weight in the feature extractor
has changed, you cannot use provided checkpoint in the original repository
to do transfer learning. You must modify those checkpoints so that the shape
of the first convolution weight match your model. It's quite easy to do this.
Bascially, you want to load all the weights of the checkpoint into memory as numpy array,
look for the first convolution weight, change it, create tf.Variable for
all the weights using the new shapes and save it back as a new checkpoint.
A good reference would be this file from TensorFlow python tools:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/inspect_checkpoint.py
