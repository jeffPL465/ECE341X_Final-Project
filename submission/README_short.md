# Jeffrey Ponce Lopez
# ECE341X Final Project

## Overview

The goal of this project was to train and deploy a person detection model using the Visual Wake Words (VWW) dataset. The model is based on the MobileNetV1 architecture, which is commonly used for lightweight computer vision tasks. The main objective was to run this model efficiently on a Raspberry Pi using TensorFlow Lite, while maintaining an accuracy of at least 80%.

The dataset used in this project contains small 96×96 RGB images labeled as either person or non_person. The training pipeline was used to train the model using deterministic dataset splits (`train.txt` for training and `val.txt` for validation). After training, the model was saved in Keras `.h5` format.

Since Raspberry Pi can't run the Keras models directly, I converted the model into TensorFlow Lite (`.tflite`) format.

To make the model more efficient for deployment, I applied a compression technique during the conversion process.


## Compression Strategy

To reduce the size of the model while keeping its performance high, post-training quantization was applied while converting to the TensorFlow Lite.

Quantization works by reducing the numerical precision of the model’s weights. Instead of storing weights using 32-bit floating point numbers (the standard format used during training), the TensorFlow Lite converter stores them using 16-bit floating point values. This reduced the amount of memory required to store the model. This was done by enabling optimization in the TensorFlow Lite converter:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

Applying this optimization reduced the model size significantly. The original TensorFlow Lite model was about 0.826 MB, and after quantization the size decreased to about 0.437 MB, which is almost reduced in half.

One advantage of quantization is that it usually maintains very similar accuracy while making the model smaller and easier to deploy on devices with limited storage and memory.


## Tradeoffs and Bottlenecks

While quantization reduces model size, it can sometimes introduce small accuracy losses because the model weights are represented with lower precision. In this project, the model stayed almost identical in terms of accuracy after quantization, which shows that MobileNet models are work well with this type of compression.

Another consideration is inference performance on the Raspberry Pi. Since the model runs on a CPU-only environment, performance can be limited by it's available processing power and memory bandwidth. In this case, smaller models are beneficial because they don't require as much memory and are easier for the CPU to process.

Although quantization reduced the model size significantly, the measured latency increased slightly. This can occur because the TensorFlow Lite interpreter might of introduced additional overhead when handling quantized operations on some hardware. Overall, the compression strategy improved the deployability of the model while maintaining strong accuracy.


## Final Model Metrics

Model: model.tflite
Accuracy: 0.8479 (84.79%)
Model size: 0.4371 MB
MACs: 7.4897 M
Latency (official p90): 5.99 ms
Peak RSS memory: 359.88 MB
Score: 0.9483


## Reproduce

1. Train the model using the training script:

```
python src/train_vww.py
```

2. Convert the trained Keras model to TensorFlow Lite with quantization enabled.

3. Evaluate the final model using the evaluation script:

```
python src/evaluate_vww.py --model trained_models/vww_96_quantized.tflite --split test_public --compute_score --export_json
```

The resulting `.tflite` model and generated `.json` metadata file were used for submission.
