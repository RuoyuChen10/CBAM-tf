import tensorflow as tf
from CBAM import *

def main():
    X = tf.placeholder(tf.float32, [None, 299, 299, 3], name="Input")
    y = CBAM(X, r=2)
    sess = tf.Session()
    writer = tf.summary.FileWriter("./logs", sess.graph)
    writer.close()


if __name__ == '__main__':
    main()