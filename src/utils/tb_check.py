import psutil


def is_tensorboard_running():
    for proc in psutil.process_iter():
        if "tensorboard" in proc.name():
            return True
    return False

def close_tensorboard_processes():
    for proc in psutil.process_iter():
        try:
            if "tensorboard" in proc.name():
                proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

