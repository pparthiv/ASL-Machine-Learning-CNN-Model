import tensorflow as tf

def create_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, (5, 5), strides = 1, padding = 'same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides = 2, padding = 'same'),
    tf.keras.layers.Conv2D(64, (3, 3), strides = 1, padding = 'same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), 2, padding = 'same'),
    tf.keras.layers.Conv2D(32, (2, 2), strides = 1, padding = 'same', activation = 'relu'),
    tf.keras.layers.MaxPooling2D((2, 2), 2, padding = 'same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(26, activation='softmax')])

  model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
  
  return model