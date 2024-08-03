import time
import subprocess
import psutil

def is_process_running(process_name):
    """检查指定的进程是否正在运行。"""
    for proc in psutil.process_iter(['pid', 'name']):
        if process_name in proc.info['name']:
            return True
    return False

def start_process():
    """启动 FastAPI 服务。"""
    return subprocess.Popen(["uvicorn", "server:app", "--host", "127.0.0.1", "--port", "6666"])

def monitor_service():
    """监控 FastAPI 服务并在必要时重启，使用指数延迟重启策略。"""
    process = start_process()
    restart_delay = 1  # 初始延迟时间为1秒
    max_delay = 300  # 最大延迟时间为5分钟

    while True:
        time.sleep(5)
        if process.poll() is not None:  # 检查进程是否已终止
            print("Service stopped. Restarting...")
            while True:
                process = start_process()
                time.sleep(restart_delay)
                if process.poll() is None:  # 服务成功启动
                    print("Service restarted successfully.")
                    restart_delay = 1  # 重置延迟时间
                    break
                else:
                    print(f"Failed to restart. Retrying in {restart_delay} seconds...")
                    restart_delay = min(restart_delay * 2, max_delay)  # 延迟时间呈指数增长

if __name__ == '__main__':
    monitor_service()
