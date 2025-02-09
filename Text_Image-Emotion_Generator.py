import subprocess
import sys
from PyQt5.QtWidgets import QApplication, QFileDialog
# 执行第一个 Python 文件并获取输出

chinese_to_english={'恐惧': 'fear', '愤怒': 'angry', '厌恶': 'disgust', '开心': 'happy', '伤心': 'sad', '惊讶': 'surprise'}
try:
    result = subprocess.run(['python', './class_predict/predict.py'], capture_output=True, text=True, check=True)
    output = result.stdout.strip()
    print(output)
    output=chinese_to_english[output]
    app = QApplication(sys.argv)
        # 弹出文件选择对话框
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName()

    print(f"你选择的文件路径是: {file_path}")

    imgpath=file_path
    # 构造参数
    parameter_emo = f"--emotion {output}"
    parameter_img = f" --imgpath {imgpath}"
    print(f"Output from file1: {output}")

    # 将构造好的参数传递给第二个 Python 文件
    subprocess.run(['python', 'generate.py'] + parameter_emo.split() + parameter_img.split(), check=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred while running Python files: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

    #    退出应用程序
    sys.exit(app.exec_())