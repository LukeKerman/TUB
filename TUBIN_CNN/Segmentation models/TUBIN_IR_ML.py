import h5py
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
#%matplotlib widget
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers.legacy import SGD, Adam
import cv2
from tqdm import tqdm

class TUBINImageProcessor:
    
    def __init__(self, path, section_size=256, overlap=0.25):
        self.path = path
        self.section_size = section_size
        self.overlap = overlap
        self.image_data, self.cloud_map_data = self.open_data()
        self.class_weights = self.define_class_weights()
        self.sq_images = self.cut_images()
        self.sq_labels = self.cut_labels()
        self.image_train, self.image_test, self.cloud_train, self.cloud_test = self.train_test_split()

    def open_data(self):
        image_data_list = []
        cloud_map_data_list = []
        norm_temp = 50

        for filename in os.listdir(self.path):
            if filename.endswith(".h5"):
                with h5py.File(os.path.join(self.path, filename), 'r') as file:
                    if 'image' in file:
                        img_data = file['image'][:] / norm_temp
                        cp_data = file['label_map'][:]
                        image_data_list.append(img_data)
                        cloud_map_data_list.append(cp_data)
                    else:
                        print(f"Dataset 'image' not found in file: {os.path.join(self.path, filename)}")

        image_data = np.array(image_data_list)
        cloud_map_data = np.array(cloud_map_data_list)
        cloud_map_data = np.squeeze(cloud_map_data, axis=-2)

        return image_data, cloud_map_data

    def cut_images(self):
        return self._cut(self.image_data)

    def cut_labels(self):
        return self._cut(self.cloud_map_data, label=True)

    def _cut(self, data, label=False):
        cut_frames = []
        overlap_step = self.section_size * (1 - self.overlap)
        for image in data:
            num_rows, num_cols = [
                int((image.shape[i] - self.section_size) // overlap_step + 1)
                for i in range(2)
            ]
            for i in range(num_rows):
                for j in range(num_cols):
                    start_row = int(i * overlap_step)
                    start_col = int(j * overlap_step)
                    cut_frames.append(image[start_row:start_row+self.section_size, start_col:start_col+self.section_size, :])

        return np.array(cut_frames)

    def define_class_weights(self):
        total_counts = np.prod(self.cloud_map_data.shape) / 4
        class_counts = np.sum(self.cloud_map_data, axis=(0, 1, 2))
        class_weights = {i: max(1, total_counts / count) for i, count in enumerate(class_counts)}

        class_weights = {
        0: 1,  # no_cloud
        1: 2,  # cloud
        2: 2,  # water
        3: 15  # fire
    }
        return class_weights

    def train_test_split(self):
        return train_test_split(self.sq_images, self.sq_labels, test_size=0.2, random_state=42)
    
    def plot_samples(self, num_samples=3):
        # Ensure there are enough samples to plot
        num_samples = min(num_samples, len(self.sq_images))
        i_offset = np.random.randint(0, len(self.sq_images) - num_samples)

        plt.close('all')
        plt.figure(figsize=(12, num_samples*5))

        for i in range(num_samples):
            # Calculate the current index with offset
            current_index = i + i_offset
            
            # Plot the image
            plt.subplot(num_samples, 2, i*2 + 1)
            plt.imshow(self.sq_images[current_index, :, :, 0], cmap='magma', vmin=-1, vmax=1)
            plt.title(f'Image - Sample {current_index + 1}')
            plt.colorbar(shrink=0.7)

            # Assuming cut_labels is one-hot encoded and needs argmax to convert to categorical labels
            # Adjust this if cut_labels has a different structure
            combined_map = np.argmax(self.sq_labels[current_index, :, :, :], axis=-1)

            # Plot the corresponding cloud map
            plt.subplot(num_samples, 2, i*2 + 2)
            plt.imshow(combined_map, cmap='binary_r')
            plt.title(f'Cloud Map - Sample {current_index + 1}')
            plt.colorbar(shrink=0.7)

        plt.tight_layout()
        plt.show()


class UNetTrainer:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.define_unet()
        self.datagen, self.labelgen = self.setup_datagen()
        self.callbacks = self.define_callbacks()
        self.history = None

    def define_unet(self):
        
        # Encoder
        inputs = layers.Input(shape=self.input_shape)
        conv1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(inputs)
        batch1 = layers.BatchNormalization()(conv1)
        conv2 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(batch1)
        batch2 = layers.BatchNormalization()(conv2)
        pool1 = layers.MaxPooling2D((2, 2))(batch2)

        conv3 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(pool1)
        batch3 = layers.BatchNormalization()(conv3)
        conv4 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(batch3)
        batch4 = layers.BatchNormalization()(conv4)
        pool2 = layers.MaxPooling2D((2, 2))(batch4)

        conv5 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(pool2)
        batch5 = layers.BatchNormalization()(conv5)
        conv6 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(batch5)
        batch6 = layers.BatchNormalization()(conv6)
        pool3 = layers.MaxPooling2D((2, 2))(batch6)

        #Bottleneck
        conv7 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(pool3)
        batch7 = layers.BatchNormalization()(conv7)
        conv8 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(batch7)

        # Decoder
        up9 = layers.UpSampling2D(size=(2, 2))(conv8)
        concat9 = layers.Concatenate()([up9, conv6])
        batch9 = layers.BatchNormalization()(concat9)
        conv9 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(batch9)
        batch10 = layers.BatchNormalization()(conv9)
        conv10 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(batch10)
        dropout10 = layers.Dropout(0)(conv10)

        up11 = layers.UpSampling2D(size=(2, 2))(dropout10)
        concat11 = layers.Concatenate()([up11, conv4])
        batch11 = layers.BatchNormalization()(concat11)
        conv11 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(batch11)
        batch12 = layers.BatchNormalization()(conv11)
        conv12 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(batch12)
        dropout12 = layers.Dropout(0.05)(conv12)

        up13 = layers.UpSampling2D(size=(2, 2))(dropout12)
        concat13 = layers.Concatenate()([up13, conv2])
        batch13 = layers.BatchNormalization()(concat13)
        conv13 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(batch13)
        batch14 = layers.BatchNormalization()(conv13)
        conv14 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(batch14)
        dropout14 = layers.Dropout(0.1)(conv14)

        # Output layer
        output = layers.Conv2D(4, (1, 1), activation='softmax')(dropout14)

        model = models.Model(inputs=inputs, outputs=output)
        optimizer = SGD(learning_rate=0.005)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        return model

    def setup_datagen(self):

        datagen = ImageDataGenerator(
            rotation_range=45, 
            width_shift_range=0.1, 
            height_shift_range=0.1,
            horizontal_flip=True, 
            vertical_flip=True, 
            fill_mode='reflect')
        labelgen = ImageDataGenerator(
            rotation_range=45, 
            width_shift_range=0.1, 
            height_shift_range=0.1,
            horizontal_flip=True, 
            vertical_flip=True, 
            fill_mode='reflect')
        
        return datagen, labelgen

    def define_callbacks(self):

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_lr=1e-4)
        checkpoint = ModelCheckpoint('model_epoch_{epoch:02d}.h5', save_best_only=False)
        return [reduce_lr]

    def fit_model(self, image_train, cloud_train, image_test, cloud_test, batch_size=8, epochs=50, class_weights=None):

        steps_per_epoch = len(image_train) // batch_size
        gen_seed = 42
        image_generator = self.datagen.flow(image_train, batch_size=batch_size, seed=gen_seed)
        mask_generator = self.labelgen.flow(cloud_train, batch_size=batch_size, seed=gen_seed)
        train_generator = zip(image_generator, mask_generator)

        self.history = self.model.fit(
            train_generator, 
            steps_per_epoch=steps_per_epoch, 
            epochs=epochs,
            class_weight=class_weights, 
            validation_data=(image_test, cloud_test),
            #callbacks=self.callbacks
            )

        return self.history
    
    def continue_training(self, image_train, cloud_train, image_test, cloud_test, additional_epochs=100, batch_size=8, class_weights=None):

        steps_per_epoch = len(image_train) // batch_size
        gen_seed = 42
        image_generator = self.datagen.flow(image_train, batch_size=batch_size, seed=gen_seed)
        mask_generator = self.labelgen.flow(cloud_train, batch_size=batch_size, seed=gen_seed)
        train_generator = zip(image_generator, mask_generator)

        # Fit the model again for additional_epochs
        additional_history = self.model.fit(
            train_generator, 
            steps_per_epoch=steps_per_epoch, 
            epochs=additional_epochs,
            class_weight=class_weights, 
            validation_data=(image_test, cloud_test),
            callbacks=self.callbacks
            )

        # Append the new history to the existing one
        self.append_history(additional_history)

    def append_history(self, additional_history):

        # Define the keys you're interested in
        keys_of_interest = ['accuracy', 'loss', 'val_accuracy', 'val_loss']
        
        if self.history is None:
            self.history = additional_history
        else:
            for key in keys_of_interest:
                self.history.history[key].extend(additional_history.history[key])

    def predict_train_data(self, data, true_labels):
        
        # Predict on data
        predictions = self.model.predict(data)
        predicted_labels = np.argmax(predictions, axis=-1)
        
        # Calculate difference
        true_labels_argmax = np.argmax(true_labels, axis=-1)
        diff = predicted_labels - true_labels_argmax
        
        return predictions, diff
    
    def visualize_train_pred(self, data, predictions, true_labels, diff, num_samples=6):
    
        cmap = mcolors.ListedColormap(['wheat', 'cornflowerblue', 'mediumblue', 'red'])
        rand_offset = random.randint(0, data.shape[0] - num_samples - 1)
        print(f"Random offset: {rand_offset}, Total samples: {data.shape[0]}")

        plt.figure(figsize=(12, 8))

        for i in range(num_samples):
            # Original image (input)
            plt.subplot(4, num_samples, i + 1)
            plt.imshow(data[i + rand_offset], cmap='magma', vmin=-1, vmax=1)
            plt.title(f'Test Image {i}', fontsize=8)
            plt.axis('off')

            # Predicted mask (output)
            plt.subplot(4, num_samples, i + 1 + num_samples)
            plt.imshow(np.argmax(predictions[i + rand_offset], axis=-1), cmap=cmap, vmin=-0.5, vmax=3)
            plt.title(f'Predicted Mask {i}', fontsize=8)
            plt.axis('off')

            # True mask
            true_labels_argmax = np.argmax(true_labels, axis=-1)
            plt.subplot(4, num_samples, i + 1 + 2 * num_samples)
            plt.imshow(true_labels_argmax[i + rand_offset], cmap=cmap, vmin=-0.5, vmax=3)
            plt.title(f'Ground Truth Mask {i}', fontsize=8)
            plt.axis('off')

            # Diff mask
            plt.subplot(4, num_samples, i + 1 + 3 * num_samples)
            plt.imshow(diff[i + rand_offset], cmap='RdYlGn', vmin=-1., vmax=1.)
            plt.title(f'Mask Diffs {i}', fontsize=8)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def plot_history(self):

        plt.close('all')
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        #plt.yticks(np.arange(0.8,1.01,0.01))
        #plt.ylim(0.89,1)
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylim(top=1)
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylim(bottom=0)
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()


class IRImageClassifier:
    def __init__(self, model_path):
        self.model = models.load_model(model_path)
        self.section_size = 256
        self.batch_size = 32  # Adjust based on your hardware capabilities
        # Initialize placeholders for image data and prediction results
        self.image_data = None
        self.class_map = None
        self.prob_map = None
        self.hotspots = None
        

    def predict_classmap(self, img):
        original_height, original_width = img.shape[:2]
        prob_map = np.zeros((original_height, original_width, 4))
        counter_map = np.zeros((original_height, original_width))
        edge = 5
        sections = []
        coordinates = []

        for y in tqdm(range(0, original_height, self.section_size // 2)):
            for x in range(0, original_width, self.section_size // 2):
                if y + self.section_size > original_height:
                    y = original_height - self.section_size
                if x + self.section_size > original_width:
                    x = original_width - self.section_size

                section = img[y:y+self.section_size, x:x+self.section_size]
                section = np.expand_dims(section, axis=-1)
                sections.append(section)
                coordinates.append((y, x))

                if len(sections) == self.batch_size or (y == original_height - self.section_size and x == original_width - self.section_size):
                    batch_sections = np.stack(sections, axis=0)
                    preds = self.model.predict(batch_sections, verbose=0)

                    for i, pred in enumerate(preds):
                        pred = np.squeeze(pred)
                        y, x = coordinates[i]

                        if edge > 0:
                            prob_map[y+edge:y+self.section_size-edge, x+edge:x+self.section_size-edge] += pred[edge:-edge, edge:-edge]
                            counter_map[y+edge:y+self.section_size-edge, x+edge:x+self.section_size-edge] += 1
                        else:
                            prob_map[y:y+self.section_size, x:x+self.section_size] += pred
                            counter_map[y:y+self.section_size, x:x+self.section_size] += 1

                    sections = []
                    coordinates = []

        # Finalizing class map and probability map
        class_map = np.argmax(prob_map, axis=-1)
        prob_map /= np.maximum(counter_map[:, :, None], 1)  # Avoid division by zero

        # Process fire hotspots
        fire_map = (class_map == 3).astype(np.uint8)
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(fire_map, connectivity=8)

        hotspots = []
        if num_labels > 1:
            hotspots = np.hstack((centroids[1:, :], stats[1:, 4].reshape(-1, 1)))

        # Set the class attributes for later use
        self.image_data = img
        self.class_map = class_map
        self.prob_map = prob_map
        self.hotspots = hotspots

        return class_map, prob_map, hotspots

    def plot_predictions(self):
        if self.prob_map is None or self.hotspots is None:
            print("No prediction data available.")
            return

        plt.close('all')
        plt.figure(figsize=(14, 6))

        # Input Image
        plt.subplot(1, 2, 1)
        plt.imshow(self.image_data, cmap='magma', vmin=-1, vmax=1)
        plt.colorbar(shrink=0.65)
        plt.title('Input Image', pad=10)

        # Output Probability Maps and Hotspots
        titles = ['Landmass', 'Cloud', 'Water', 'Fire']
        cmaps = ['Greens', 'Purples', 'Blues', 'Oranges']
        vminmax = [(0.9, 1.05), (0, 1.1), (0.3, 1.2), (0, 0.5)]

        for i, (title, cmap, vmin, vmax) in enumerate(zip(titles, cmaps, *zip(*vminmax))):
            if i >= 2:
                ax = plt.subplot(2, 4, i + 5)
            else:
                ax = plt.subplot(2, 4, i + 3)
            plt.imshow(self.prob_map[:, :, i], cmap=cmap, vmin=vmin, vmax=vmax)
            plt.title(f'Output {title} (Pmax = {np.max(self.prob_map[:, :, i])*100:.2f}%)', fontsize=9)
            if title == 'Fire':  # Plot hotspots for the fire class
                for x, y, area in self.hotspots:
                    radius = 5 + 3*np.sqrt(area)
                    circle = Circle((x, y), radius, color='orangered', fill=False, linewidth=1, alpha=0.5)
                    ax.add_patch(circle)

        plt.show()

    def create_thresholded_composite_map(self):
        if self.prob_map is None or self.class_map is None:
            print("No prediction data available.")
            return

        # Define a custom colormap for the class labels
        cmap = mcolors.ListedColormap(['white', 'lightgreen', 'cornflowerblue', 'mediumblue', 'red'])

        # Initialize the class label image based on thresholds
        class_labels_image = np.zeros(self.prob_map.shape[:2], dtype=int)

        # Assign class labels based on thresholds
        class_labels_image[np.where(self.prob_map[:,:,0] > 0.95)] = 1  # Landmass
        class_labels_image[np.where(self.prob_map[:,:,1] > 0.25)] = 2  # Cloud
        class_labels_image[np.where(self.prob_map[:,:,2] > 0.35)] = 3   # Water
        class_labels_image[np.where(self.class_map == 3)] = 4         # Fire

        unique_classes = np.unique(class_labels_image)
        class_colors = ['white', 'lightgreen', 'cornflowerblue', 'mediumblue', 'red']  # Original colors
        dynamic_colors = [class_colors[i] for i in unique_classes]
        cmap = mcolors.ListedColormap(dynamic_colors)

        # Create a figure to display the composite map
        plt.figure(figsize=(12, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(self.image_data, cmap='magma', vmin=-1, vmax=1)
        plt.title('Input Image', pad=10)

        plt.subplot(1, 2, 2)
        plt.imshow(class_labels_image, cmap=cmap)

        # Overlay hotspots if any
        if self.hotspots is not None and len(self.hotspots) > 0:
            ax = plt.gca()  # Get current axes
            for x, y, area in self.hotspots:
                radius = 5 + 3*np.sqrt(area)  # Example radius calculation
                circle = Circle((x, y), radius, color='orangered', fill=False, linewidth=1, alpha=0.5)
                ax.add_patch(circle)

        #plt.axis('off')  # Hide axes
        plt.title('Composite Output Map with Thresholds and Hotspots')
        plt.show()