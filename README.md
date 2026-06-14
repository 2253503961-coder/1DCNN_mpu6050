# 1DCNN_mpu6050 — Fall Detection with MPU6050 and 1D CNN

A lightweight fall-detection pipeline using a 6-axis IMU sensor (MPU6050) and a 1D convolutional neural network. Designed for edge deployment on resource-constrained microcontrollers (STM32F103 / Arduino / ESP32), the project covers the full workflow: **sensor calibration → auto-labeled data collection → model training → ONNX + pure-C export**.

## Features

- **Zero-bias calibration** — static calibration of accelerometer and gyroscope offsets, saved to `mpu_offsets.json`
- **Auto-labeled data collection** — real-time SVM (Signal Vector Magnitude) threshold-based impact detection that automatically labels fall events (1) vs. normal activity (0), including a pre-fall ring buffer
- **Ultra-lightweight 1D CNN** — only 4-channel Conv1d + MaxPool + a single FC layer; 6×50 input, 2-class output
- **ONNX export** — model saved as `tiny_cnn.onnx` for visualization, validation, or STM32Cube.AI import
- **Pure-C header generation** — all weights, biases, and normalization parameters exported to `cnn_weights.h` for hand-written inference on bare-metal MCUs
- **Serial debug tool** — `serial_test.py` for raw serial monitoring / troubleshooting

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.7+ |
| Deep Learning | PyTorch |
| Serial Communication | pySerial |
| Data Processing | NumPy, Pandas |
| Model Export | ONNX (opset 11) |
| Target Hardware | STM32F103 / Arduino / ESP32 + MPU6050 |

## Directory Structure

```
1DCNN_mpu6050/
├── train.py                  # Model training + ONNX export + C header generation
├── mpu_serial.py             # Data pipeline: calibration & auto-labeled collection
├── serial_test.py            # Raw serial monitor for debugging
├── my_dataset.csv            # Collected IMU dataset (ax,ay,az,gx,gy,gz,label)
├── mpu_offsets.json          # Saved calibration offsets
├── tiny_cnn.onnx             # Exported ONNX model
├── fall_detection_f103.onnx  # STM32F103-optimized ONNX model
├── cnn_weights.h             # Auto-generated C header with all weights
├── 实验1：基于MPU6050的跌倒检测数据采集与自动标注实验指导书.md
├── 实验2：MPU6050跌倒检测数据集标准化采集实验指导书.md
└── 实验3：基于1D CNN的跌倒检测模型训练与边缘部署导出实验指导书.md
```

## Installation

```bash
pip install torch pyserial pandas numpy
```

## Usage

### Step 1 — Calibrate the sensor

Place the MPU6050 flat on a stable surface and run:

```bash
python mpu_serial.py --mode calib --port COM3 --baud 115200
```

This generates `mpu_offsets.json` with the static offset values.

### Step 2 — Collect data with auto-labeling

```bash
python mpu_serial.py --mode collect --port COM3 --baud 115200 --file my_dataset.csv --duration 300
```

The script automatically labels fall events (SVM > 1900) vs. normal activity in real time. Duration is in seconds; adjust `IMPACT_THRESHOLD` inside the script for your sensor's sensitivity.

### Step 3 — Train the model

```bash
python train.py
```

Outputs:
- `tiny_cnn.onnx` — ONNX model for STM32Cube.AI
- `cnn_weights.h` — Pure-C weight array for manual inference

### Step 4 — Deploy to MCU

Copy `cnn_weights.h` into your embedded C project. Implement the forward pass using the architecture described in Experiment Guide 3. Alternatively, import `tiny_cnn.onnx` directly into STM32Cube.AI.

### Debug Serial Output

If the sensor isn't sending data as expected, use the serial monitor to inspect raw output:

```bash
python serial_test.py   # Edit TARGET_PORT inside the script
```

## Model Architecture

```
Input: [Batch, 6 channels, 50 time-steps]
  ↓ Conv1d(6→4, kernel=3) → ReLU → MaxPool1d(2)
  ↓ Flatten → Linear(96→2)
Output: 2-class logits (0=normal, 1=fall)
```

**Total parameters:** fewer than 400 weights — easily fits in MCU flash.

## API / Script Reference

### `mpu_serial.py`

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mode` | str | (required) | `calib` or `collect` |
| `--port` | str | `COM3` | Serial port name |
| `--baud` | int | `115200` | Baud rate |
| `--file` | str | `dataset.csv` | Output CSV (collect mode) |
| `--duration` | int | `300` | Collection duration in seconds (collect mode) |

### Expected Serial Format

The MCU must output data in the format:

```
accel.x=1234, accel.y=-56, accel.z=16000, gyro.x=-220, gyro.y=-50, gyro.z=-190
```

Raw values are parsed by regex; units depend on your MCU's MPU6050 driver scaling.

## Notes

- The `IMPACT_THRESHOLD` (default 1900 in `mpu_serial.py`) must be tuned to your sensor's full-scale range and mounting orientation.
- `fall_detection_f103.onnx` is a pre-exported model optimized for STM32F103 deployment.
- The `.vscode/` and `__pycache__/` directories contain IDE/bytecode artifacts and are excluded from version control via `.gitignore`.
- Ensure no other application (serial monitor, Arduino IDE) is occupying the COM port before running scripts.

## License

This project is provided for educational and research purposes. Adapt the hardware configuration and threshold parameters to your specific setup before deployment.
