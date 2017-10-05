# Create a tensorflow queue over 3 devices as below
# Queues can only be created in multithread environment which is created using coordinator  and a queue runner in tensorflow

import tensorflow as tf
import numpy as np

# create queue with length = 50
def createqueue(job_index):
    queue = tf.FIFOQueue(50,tf.float32)
    return queue


#create a random vector of 50 samples to enqueue into the shared queue
randi = tf.random_normal(shape=[50,1],mean=0)
queue = createqueue(3)
enqueue_operation = queue.enqueue_many(randi)
queues = queue.dequeue()
Qurun = tf.train.QueueRunner(queue,[enqueue_operation]*2)
#visualize using tensorboard
Writer = tf.summary.FileWriter("home/skay/queue_example")
# Now setup a queue runner and a coordinator
with tf.Session() as sess:
          Writer.add_graph(sess.graph)
          coord= tf.train.Coordinator()
          #this creates threads
          enqu = Qurun.create_threads(sess,coord,start=True)
          for step in range(50):
              print(step)
              if coord.should_stop():
                   break
              #run the operations of enque
              r = sess.run(queues)
          coord.request_stop()
          coord.join(enqu)






