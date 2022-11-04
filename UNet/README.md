# UNet

### Decode the image

```
python conversion.py
```

### Train the model

```
python CityScapes_epistemic_and_aleatoric.py
```

The model will be saved as *model.h5* and if you don't want to train the model again, you can skip this step.

### 5 Examples

```
python testResult.py
```



## Results

In the calculation of epistemic uncertainty, the results after dropout of the same image are too similar, resulting in very small variance results, almost no difference, and therefore poor visualization results.

The following are original images,  semantic segmentation, aleatoric uncertainty and epistemic uncertainty.

![a3ba5019a6205c785216688ccfba1c4](UNet.assets/a3ba5019a6205c785216688ccfba1c4.png)