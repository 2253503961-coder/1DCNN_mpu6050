import serial
import time

def simple_serial_monitor(port='COM3', baudrate=115200):
    print(f"正在尝试连接串口 {port} (波特率: {baudrate})...")
    
    try:
        # 打开串口
        ser = serial.Serial(port, baudrate, timeout=1)
        print("✅ 串口连接成功！正在监听数据... (按 Ctrl+C 退出)\n")
        print("-" * 60)
        
        while True:
            # 检查串口缓冲区是否有数据
            if ser.in_waiting > 0:
                # 读取一行数据，并进行解码和去除首尾空白字符
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # 如果读取到的不为空，则打印出来
                if line:
                    # 这里你还可以加入 time.strftime("%H:%M:%S") 打印时间戳
                    print(f">> {line}")
                    
    except serial.SerialException as e:
        print(f"\n❌ 串口出错: {e}")
        print("排查建议：")
        print("1. 检查串口号是否写错（Windows 通常是 COMx，Linux/Mac 通常是 /dev/ttyUSBx）")
        print("2. 检查串口是否被其他软件（如串口调试助手、Putty）占用")
        print("3. 确认是否已正确安装了库：pip install pyserial")
        
    except KeyboardInterrupt:
        # 捕捉 Ctrl+C 退出信号，优雅地关闭串口
        print("\n🛑 检测到退出指令，已停止监听。")
        
    finally:
        # 确保程序退出时串口被安全关闭
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("🔒 串口已释放。")

if __name__ == '__main__':
    # ==========================================
    # 在这里修改为你的实际串口号和波特率
    # ==========================================
    TARGET_PORT = 'COM4'   
    TARGET_BAUD = 115200   
    
    simple_serial_monitor(port=TARGET_PORT, baudrate=TARGET_BAUD)