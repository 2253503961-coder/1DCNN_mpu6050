import serial
import re
import csv
import time
import math
import collections
import argparse
import json
import os

# 配置文件路径
OFFSET_FILE = "mpu_offsets.json"

def parse_mpu6050_line(line):
    """解析串口数据流"""
    pattern = r"accel\.x\s*=\s*(-?\d+),?\s*accel\.y\s*=\s*(-?\d+),?\s*accel\.z\s*=\s*(-?\d+),?\s*gyro\.x\s*=\s*(-?\d+),?\s*gyro\.y\s*=\s*(-?\d+),?\s*gyro\.z\s*=\s*(-?\d+)"
    match = re.search(pattern, line)
    if match:
        return [float(x) for x in match.groups()]
    return None

def calculate_svm(ax, ay, az):
    """计算合加速度"""
    return math.sqrt(ax**2 + ay**2 + az**2)

# ==========================================
# 模块 1: 零偏校准 (Calibration)
# ==========================================
def run_calibration(port, baudrate, samples=500):
    print(f"[{time.strftime('%H:%M:%S')}] 正在连接串口 {port} 进行校准...")
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
    except Exception as e:
        print(f"串口错误: {e}\n请检查是否被占用或库未正确安装。")
        return

    print("\n" + "="*50)
    print("!!! 请将设备水平放置在桌面上，并保持绝对静止 !!!")
    print("="*50)
    for i in range(3, 0, -1):
        print(f"校准将在 {i} 秒后开始...")
        time.sleep(1)

    print("\n开始采集环境底噪，请勿触碰桌面...")
    
    accel_x, accel_y, gyro_x, gyro_y, gyro_z = [], [], [], [], []
    count = 0
    
    while count < samples:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            data = parse_mpu6050_line(line)
            if data:
                ax, ay, az, gx, gy, gz = data
                accel_x.append(ax)
                accel_y.append(ay)
                # Z 轴包含重力，通常不作静态归零，故省略 az
                gyro_x.append(gx)
                gyro_y.append(gy)
                gyro_z.append(gz)
                
                count += 1
                if count % 100 == 0:
                    print(f"已采集 {count}/{samples} 个样本...")

    offsets = {
        "ax_offset": sum(accel_x) / samples,
        "ay_offset": sum(accel_y) / samples,
        "gx_offset": sum(gyro_x) / samples,
        "gy_offset": sum(gyro_y) / samples,
        "gz_offset": sum(gyro_z) / samples
    }

    with open(OFFSET_FILE, 'w') as f:
        json.dump(offsets, f, indent=4)

    print("\n✅ 校准完成！零偏数据已自动保存至 mpu_offsets.json")
    for k, v in offsets.items():
        print(f"   {k}: {v:.1f}")
    ser.close()

# ==========================================
# 模块 2: 自动标注采集 (Data Collection)
# ==========================================
def run_collection(port, baudrate, output_file, duration):
    # 1. 尝试加载校准文件
    offsets = {"ax_offset": 0, "ay_offset": 0, "gx_offset": 0, "gy_offset": 0, "gz_offset": 0}
    if os.path.exists(OFFSET_FILE):
        with open(OFFSET_FILE, 'r') as f:
            offsets = json.load(f)
        print(f"✅ 成功加载校准配置: {offsets}")
    else:
        print("⚠️ 未找到 mpu_offsets.json，将使用未校准的原始数据。建议先运行 --mode calib")

    # 2. 连接串口
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
    except Exception as e:
        print(f"串口错误: {e}")
        return

    # 3. 初始化 CSV
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'label'])

        # --- 核心参数配置 ---
        IMPACT_THRESHOLD = 1900  # 撞击阈值，根据你的设备量程调整
        BUFFER_SIZE = 50         # 跌倒前置数据 (1秒)
        POST_FALL_FRAMES = 50    # 跌倒后置数据 (1秒)
        
        buffer = collections.deque(maxlen=BUFFER_SIZE)
        is_falling = False
        post_fall_counter = 0
        fall_event_count = 0
        normal_data_skip = 0     

        print("\n" + "="*50)
        print(">>> 采集已启动！请开始执行测试动作。")
        print("="*50)
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                raw_data = parse_mpu6050_line(line)
                
                if raw_data:
                    # 施加校准补偿
                    calib_data = [
                        raw_data[0] - offsets['ax_offset'],
                        raw_data[1] - offsets['ay_offset'],
                        raw_data[2], # Z 轴保留重力
                        raw_data[3] - offsets['gx_offset'],
                        raw_data[4] - offsets['gy_offset'],
                        raw_data[5] - offsets['gz_offset']
                    ]
                    
                    ax, ay, az, gx, gy, gz = calib_data
                    svm = calculate_svm(ax, ay, az)
                    current_label = 0 
                    
                    if not is_falling:
                        if svm > IMPACT_THRESHOLD:
                            # 触发跌倒
                            is_falling = True
                            current_label = 1
                            post_fall_counter = POST_FALL_FRAMES
                            fall_event_count += 1
                            print(f"\n[!] 触发跌倒事件 #{fall_event_count} (SVM: {svm:.1f})")
                            
                            # 写入历史缓冲数据 (跌倒前兆)
                            while len(buffer) > 0:
                                writer.writerow(buffer.popleft() + [1])
                            writer.writerow(calib_data + [1]) 
                        else:
                            # 日常活动
                            current_label = 0
                            buffer.append(calib_data) 
                            
                            normal_data_skip += 1
                            if normal_data_skip % 5 == 0:
                                writer.writerow(calib_data + [0])
                    else:
                        # 跌倒中/跌倒后
                        current_label = 1
                        writer.writerow(calib_data + [1]) 
                        post_fall_counter -= 1
                        
                        if post_fall_counter <= 0:
                            is_falling = False
                            buffer.clear() 
                            print("    [✓] 跌倒动作录制完毕，恢复日常监测。\n")
                            
                    # 实时终端显示
                    print(f"监测中 -> ax:{ax:>6.0f} ay:{ay:>6.0f} az:{az:>6.0f} | gx:{gx:>6.0f} gy:{gy:>6.0f} gz:{gz:>6.0f} | SVM:{svm:>7.1f} | 标签: {current_label}")

        print(f"\n[{time.strftime('%H:%M:%S')}] 采集结束！共自动捕获 {fall_event_count} 次跌倒。数据已保存至 {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MPU6050 边缘计算数据流水线: 校准与自动标注")
    parser.add_argument('--mode', type=str, required=True, choices=['calib', 'collect'], help="calib: 执行零偏校准 | collect: 开始采集数据")
    parser.add_argument('--port', type=str, default='COM3', help="串口号 (如 COM3)")
    parser.add_argument('--baud', type=int, default=115200, help="波特率")
    parser.add_argument('--file', type=str, default='dataset.csv', help="数据保存的文件名 (仅 collect 模式)")
    parser.add_argument('--duration', type=int, default=300, help="采集持续时间，单位秒 (仅 collect 模式)")
    
    args = parser.parse_args()
    
    if args.mode == 'calib':
        run_calibration(args.port, args.baud)
    elif args.mode == 'collect':
        run_collection(args.port, args.baud, args.file, args.duration)