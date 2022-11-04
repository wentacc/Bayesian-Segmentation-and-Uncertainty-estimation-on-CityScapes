import tensorflow as tf

def read_png(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_png(img, channels=3)
    return img


def read_png_label(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_png(img, channels=1)
    return img


# ### Data Augmentation

def rand_crop(img, label):
    concat_img = tf.concat([img, label], axis=-1)
    concat_img = tf.image.resize(concat_img, [280, 560], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    crop_img = tf.image.random_crop(concat_img, [256, 256, 4])
    return crop_img[:, :, :3], crop_img[:, :, 3:]


def norm(img, label):
    img = tf.cast(img, tf.float32) / 127.5 - 1
    label = tf.cast(label, tf.int32)
    return img, label


def load_img_train(img, label):
    img = read_png(img)
    label = read_png_label(label)

    img, label = rand_crop(img, label)

    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        label = tf.image.flip_left_right(label)
    return norm(img, label)


def load_img_val(img, label):
    img = read_png(img)
    label = read_png_label(label)

    img = tf.image.resize(img, [256, 256])
    label = tf.image.resize(label, [256, 256])
    return norm(img, label)