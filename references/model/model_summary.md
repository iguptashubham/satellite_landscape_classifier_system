# Model Summary

#### Overview
The `BuildModel` function constructs a Convolutional Neural Network (CNN) using the MobileNetV2 architecture, followed by custom fully connected layers for classification. The model is designed to classify satellite images into four categories: `cloudy`, `desert`, `green_area`, and `water`.

#### Architecture
1. **MobileNetV2 Backbone**:
   - **Weights**: Pre-trained on ImageNet.
   - **Top Layer**: Excluded (`include_top=False`).
   - **Input Shape**: `(255, 255, 3)`.
   - **Trainable**: False (all layers are frozen).

2. **Custom Fully Connected Layers**:
   - **Flatten Layer**: Converts the output of MobileNetV2 to a 1D array.
   - **Dense Layer 1**: 1028 units, ReLU activation, followed by Batch Normalization.
   - **Dense Layer 2**: 504 units, ReLU activation, followed by Batch Normalization.
   - **Dense Layer 3**: 100 units, ReLU activation, followed by Batch Normalization.
   - **Dropout Layer**: 20% dropout rate for regularization.
   - **Output Layer**: 4 units (number of classes), Softmax activation for multi-class classification.

### Detailed Layer Summary
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
mobilenetv2_1.00_224 (Model) (None, 8, 8, 1280)        2257984   
_________________________________________________________________
flatten (Flatten)            (None, 81920)             0         
_________________________________________________________________
dense (Dense)                (None, 1028)              84186688  
_________________________________________________________________
batch_normalization (BatchNo (None, 1028)              4112      
_________________________________________________________________
dense_1 (Dense)              (None, 504)               518616    
_________________________________________________________________
batch_normalization_1 (Batch (None, 504)               2016      
_________________________________________________________________
dense_2 (Dense)              (None, 100)               50500     
_________________________________________________________________
batch_normalization_2 (Batch (None, 100)               400       
_________________________________________________________________
dropout (Dropout)            (None, 100)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 4)                 404       
=================================================================
Total params: 87,297,720
Trainable params: 84,715,748
Non-trainable params: 2,581,972
_________________________________________________________________
```

### Summary Explanation
- **Total Parameters**: 87,297,720 parameters in the model.
- **Trainable Parameters**: 84,715,748 parameters in the custom dense layers.
- **Non-Trainable Parameters**: 2,581,972 parameters in the MobileNetV2 base model (frozen layers).

### Usage
To print the model summary, simply call the `BuildModel` function:
```python
model = BuildModel()
```