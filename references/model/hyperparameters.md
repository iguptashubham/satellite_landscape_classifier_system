## Hyperparameters

### Model Architecture
- **Base Model**: MobileNetV2
  - **Pre-trained Weights**: ImageNet
  - **Include Top**: False
  - **Input Shape**: (255, 255, 3)
- **Custom Layers**:
  - **Flatten Layer**: Converts the output of MobileNetV2 to a 1D array.
  - **Dense Layers**:
    - Dense Layer 1: 1028 units, ReLU activation
    - Dense Layer 2: 504 units, ReLU activation
    - Dense Layer 3: 100 units, ReLU activation
  - **Batch Normalization**: Applied after each Dense layer.
  - **Dropout**: 20% rate before the final output layer.
  - **Output Layer**: 4 units, Softmax activation.

### Training Configuration
- **Optimizer**: Adam
  - **Learning Rate**: Default (0.001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Epochs**: 108 (with early stopping)

### Early Stopping
- **Monitor**: Validation Loss (`val_loss`)
- **Patience**: 3 epochs
- **Restore Best Weights**: True

### Data Augmentation
- **Rescale**: 1./255 (Normalize pixel values to [0, 1])
- **Rotation Range**: 90 degrees
- **Width Shift Range**: 30% of the image width
- **Height Shift Range**: 30% of the image height
- **Brightness Range**: (0.1, 0.4)
- **Shear Range**: 0.4
- **Zoom Range**: 0.5
- **Fill Mode**: Nearest
- **Horizontal Flip**: True
- **Vertical Flip**: True
