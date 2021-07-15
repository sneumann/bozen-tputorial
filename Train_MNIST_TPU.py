#Import Libraries and dataset
import tensorflow as tf

import os
import tensorflow_datasets as tfds

# Use matplot to plot the graphs
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

'''
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy()
'''

#Initialize the TPU cluster
#tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="node-1")
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
print("Running on TPU ", tpu.master())

tf.config.experimental_connect_to_cluster(tpu)

# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(tpu)

print("All devices: ", tf.config.list_logical_devices('TPU'))

# Initialize the strategy
strategy = tf.distribute.TPUStrategy(tpu)


# Define the network - Here a simple CNN is used for classification
def create_model():
	return tf.keras.Sequential(
			[tf.keras.layers.Conv2D(256, 3, activation='relu', input_shape=(28, 28, 1)),
			 tf.keras.layers.Conv2D(256, 3, activation='relu'),
			 tf.keras.layers.Flatten(),
			 tf.keras.layers.Dense(256, activation='relu'),
			 tf.keras.layers.Dense(128, activation='relu'),
			 tf.keras.layers.Dense(10)])


# Define the dataset and dataset pipeline

def get_dataset(batch_size, is_training=True):
	split = 'train' if is_training else 'test'
	dataset, info = tfds.load(name='mnist', split=split, with_info=True,
														as_supervised=True, try_gcs=True)

	# Normalize the input data.
	def scale(image, label):
		image = tf.cast(image, tf.float32)
		image /= 255.0
		return image, label

	dataset = dataset.map(scale)

	# Only shuffle and repeat the dataset in training. The advantage of having an
	# infinite dataset for training is to avoid the potential last partial batch
	# in each epoch, so that you don't need to think about scaling the gradients
	# based on the actual batch size.
	if is_training:
		dataset = dataset.shuffle(10000)
		dataset = dataset.repeat()

	dataset = dataset.batch(batch_size)

	return dataset


Epochs = 10
batch_size = 128
steps_per_epoch = 60000 // batch_size
validation_steps = 10000 // batch_size


# For Keras model training
'''
with strategy.scope():
	model = create_model()
	model.compile(optimizer='adam',
								loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
								metrics=['sparse_categorical_accuracy'])

train_dataset = get_dataset(batch_size, is_training=True)
test_dataset = get_dataset(batch_size, is_training=False)

model.fit(train_dataset,
					epochs=5,
					steps_per_epoch=steps_per_epoch,
					validation_data=test_dataset,
					validation_steps=validation_steps)
'''

# Training the model inside a custom training loop

# Create the model, optimizer and metrics inside the strategy scope, so that the
# variables can be mirrored on each device.
with strategy.scope():
	model = create_model()
	optimizer = tf.keras.optimizers.Adam()
	training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
	training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
			'training_accuracy', dtype=tf.float32)


# Calculate per replica batch size, and distribute the datasets on each TPU
# worker.

per_replica_batch_size = batch_size // strategy.num_replicas_in_sync

# Distribute the dataset
train_dataset = strategy.experimental_distribute_datasets_from_function(lambda _: get_dataset(per_replica_batch_size, is_training=True))

@tf.function
def train_step(iterator):
	"""The step function for one training step."""

	def step_fn(inputs):
		"""The computation to run on each TPU device."""
		images, labels = inputs
		with tf.GradientTape() as tape:
			logits = model(images, training=True)
			loss = tf.keras.losses.sparse_categorical_crossentropy(
					labels, logits, from_logits=True)
			loss = tf.nn.compute_average_loss(loss, global_batch_size=batch_size)
		grads = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
		training_loss.update_state(loss * strategy.num_replicas_in_sync)
		training_accuracy.update_state(labels, logits)

	strategy.run(step_fn, args=(next(iterator),))


# Run the training loop
steps_per_eval = 10000 // batch_size

train_iterator = iter(train_dataset)

# Arrays to store calculated statistics for plots
loss_history = []
training_accuracy_plot = []

'''
# Create path for training checkpoints
checkpoint_path = "path/to/checkpoint"
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
    start_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])

'''

for epoch in range(Epochs):
	print('Epoch: {}/10'.format(epoch))

	for step in range(steps_per_epoch):
		train_step(train_iterator)
	print('Current step: {}, training loss: {}, accuracy: {}%'.format(
			optimizer.iterations.numpy(),
			round(float(training_loss.result()), 4),
			round(float(training_accuracy.result()) * 100, 2)))
	loss_history.append(round(float(training_loss.result()), 4))
	training_accuracy_plot.append(round(float(training_accuracy.result()), 2))
	training_loss.reset_states()
	training_accuracy.reset_states()
	model.save_weights("MNIST_TPU_model_"+str(epoch)+"_.h5")

#Matplot plot depiction
plt.subplot(2,1,1)
plt.plot(loss_history, '-o', label='Loss value')
plt.title('Training Loss')
plt.xlabel('Epoch x Batches')
plt.ylabel('Loss Value')
plt.legend(ncol=2, loc='upper right')
plt.subplot(2,1,2)
plt.gca().set_ylim([0,1.0])
plt.plot(training_accuracy_plot, '-o', label='Train Accuracy value')
plt.title('Train Accuracy')
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')
plt.gcf().set_size_inches(20, 20)
plt.savefig("Lossplot_.jpg")
plt.close()
