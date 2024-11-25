import pickle, os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# Open the file in read-binary mode
with open(r'satellite_landscape_classification\reports\info.pkl', 'rb') as file:
    info = pickle.load(file)

#Accuracy

plt.plot(info.history['accuracy'], label = 'train')
plt.plot(info.history['val_accuracy'], label = 'validation')
plt.title(f"Train Accuracy vs Validation accuracy | epochs - {len(info.history['accuracy'])}")
plt.legend()
plt.savefig(os.path.join(r'satellite_landscape_classification\reports\figures','accuracy_vs_val_accuracy.png'))
plt.show()

plt.plot(info.history['loss'], label = 'train')
plt.plot(info.history['val_loss'], label = 'validation')
plt.title(f"Train loss vs Validation loss | epochs - {len(info.history['loss'])}")
plt.legend()
plt.savefig(os.path.join(r'satellite_landscape_classification\reports\figures','loss_vs_val_loss.png'))
plt.show()