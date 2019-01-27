import os
import numpy as np
import tensorflow as tf

from argparse import ArgumentParser
from tensorflow.contrib.tensorboard.plugins import projector


def load_vectors(vectors_file_path):
    return np.loadtxt(vectors_file_path)


def project(log_dir, vectors_file_path, metadata_file_path, sprite_sheet_file_path=None):
    vectors = load_vectors(vectors_file_path)
    NAME_TO_VISUALISE_VARIABLE = "vectors"
    vectors = np.reshape(vectors, newshape=(len(vectors), vectors.shape[1]))

    embedding_var = tf.Variable(vectors, name=NAME_TO_VISUALISE_VARIABLE)
    summary_writer = tf.summary.FileWriter(log_dir)
    config = projector.ProjectorConfig()

    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Specify where you find the metadata
    embedding.metadata_path = metadata_file_path  # 'metadata.tsv'

    if sprite_sheet_file_path:
        # Specify where you find the sprite (we will create this later)
        embedding.sprite.image_path = sprite_sheet_file_path  # 'mnistdigits.png'
        embedding.sprite.single_image_dim.extend([28, 28])

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(log_dir, "model.ckpt"), 1)


def main(log_dir, vectors_file_path, metadata_file_path, sprite_sheet_file_path):
    project(log_dir, vectors_file_path, metadata_file_path, sprite_sheet_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_dir", required=False, default="/tmp/logs/")
    parser.add_argument("--vectors_file_path", required=True)
    parser.add_argument("--metadata_file_path", required=True)
    parser.add_argument("--sprite_file_path", required=False, default=None)

    args = parser.parse_args()
    main(args.log_dir, args.vectors_file_path, args.metadata_file_path, args.sprite_file_path)
