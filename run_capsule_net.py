import tensorflow as tf
import numpy as np
import Capsule.capsule_net as capsule
import Capsule.utils as utl


def run_capsule_net():
    a_capsule = capsule.CapsuleNet(batch_size=64, input_size=[28, 28, 1], class_num=10, training_epochs=35,
                                   check_step=256, learning_rate=0.001)
    # a_capsule.train_net()
    # a_capsule.test_net(batch_num=1)
    # a_capsule.test_format_generate()


if __name__ == '__main__':
    run_capsule_net()