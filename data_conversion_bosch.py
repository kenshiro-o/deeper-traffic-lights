import tensorflow as tf
import yaml
import os
from tqdm import tqdm
from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('input_yaml', '', 'Path to input yaml file')
flags.DEFINE_boolean('test_dataset', False, 'Whether we are generating records for the test dataset')
FLAGS = flags.FLAGS

LABEL_DICT =  {
    "Green" : 1,
    "Red" : 2,
    "GreenLeft" : 3,
    "GreenRight" : 4,
    "RedLeft" : 5,
    "RedRight" : 6,
    "Yellow" : 7,
    "off" : 8,
    "RedStraight" : 9,
    "GreenStraight" : 10,
    "GreenStraightLeft" : 11,
    "GreenStraightRight" : 12,
    "RedStraightLeft" : 13,
    "RedStraightRight" : 14
    }

def create_tf_example(example):
    
    # Bosch
    height = 720 # Image height
    width = 1280 # Image width

    filename = example['path'] # Filename of the image. Empty if image is not from file
    filename = filename.encode()

    with tf.gfile.GFile(example['path'], 'rb') as fid:
        encoded_image = fid.read()

    image_format = 'png'.encode() 

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for box in example['boxes']:
        #if box['occluded'] is False:        
        xmins.append(float(box['x_min'] / width))
        xmaxs.append(float(box['x_max'] / width))
        ymins.append(float(box['y_min'] / height))
        ymaxs.append(float(box['y_max'] / height))
        classes_text.append(box['label'].encode())
        classes.append(int(LABEL_DICT[box['label']]))


    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def main(_):    
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    
    examples = yaml.load(open(FLAGS.input_yaml, 'rb').read())
    
    len_examples = len(examples)
    print("Loaded ", len(examples), "examples")

    print("Rewriting paths...")
    for i in range(len(examples)):
        if FLAGS.test_dataset:
            # Paths for the test dataset are incorrect - so we are making sure to fix this
            file_name = examples[i]['path'].split("/")[-1]
            examples[i]['path'] = str(os.path.abspath(os.path.join(os.path.dirname(FLAGS.input_yaml), 'rgb/test/' + file_name)))
        else:
            examples[i]['path'] = str(os.path.abspath(os.path.join(os.path.dirname(FLAGS.input_yaml), examples[i]['path'])))
    
    print("Creating TF records...")
    for example in tqdm(examples):
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())       

    writer.close()

if __name__ == '__main__':
    tf.app.run()
