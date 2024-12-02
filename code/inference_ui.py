import sys
import os
import time
import torch
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QFrame,
)
from PyQt5.QtCore import Qt
from utils.ui import InfoShowWidget, AirBtn, FileSelector, FolderSelector
import torchaudio
from asteroid.losses import PITLossWrapper
from asteroid.losses.sdr import PairwiseNegSDR


class InferenceInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("降噪模块")

        # Set window size to 2/3 of the screen and center it
        screen = QApplication.desktop().screenGeometry()
        width, height = screen.width() * 2 // 3, screen.height() * 2 // 3
        self.setGeometry(
            (screen.width() - width) // 2,
            (screen.height() - height) // 2,
            width,
            height,
        )

        # Layouts
        main_layout = QVBoxLayout()
        model_input_layout = QHBoxLayout()
        result_show_layout = QVBoxLayout()

        # First Module: Input Parameters
        self.ckp_selector = FileSelector(
            default_file=r"C:\Users\june\Workspace\Asteroid\checkpoint\DPRNN-bs2_epoch30_sr125000.pth",
            selector_text="未选择模型文件:",
            btn_text="选择",
            filetype="PyTorch Model Files (*.pth)",
        )
        load_model_button = AirBtn(
            "加载", fixed_size=(100, 50), background_color="#13a460"
        )
        load_model_button.clicked.connect(self.load_model)

        self.audio_file_selector = FileSelector(
            selector_text="未选择音频文件:",
            btn_text="选择",
            filetype="Audio Files (*.wav)",
        )
        audio_enhance_button = AirBtn(
            "降噪", fixed_size=(100, 50), background_color="#13a460"
        )
        audio_enhance_button.clicked.connect(self.audio_enhance)

        self.output_folder_selector = FolderSelector(
            selector_text="未选择输出文件夹:",
            default_folder=r"C:\Users\june\Workspace\Asteroid\code\output",
            btn_text="选择",
        )

        model_input_layout.addWidget(self.output_folder_selector)
        model_input_layout.addWidget(self.ckp_selector)
        model_input_layout.addWidget(load_model_button)
        model_input_layout.addWidget(self.audio_file_selector)
        model_input_layout.addWidget(audio_enhance_button)

        # Add a line separator
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)

        # Second Module: Result Show

        # Add a line separator

        self.result_sisnr_widget = InfoShowWidget(
            info_label="SI-SNR", label_bg_color="#000000"
        )
        self.result_time_widget = InfoShowWidget(
            info_label="耗时", label_bg_color="#000000"
        )

        first_line = QHBoxLayout()
        first_line.addWidget(self.result_sisnr_widget)
        first_line.addWidget(self.result_time_widget)

        self.res_output_path_widget = InfoShowWidget(
            info_label="输出文件", label_bg_color="#000000"
        )
        self.res_output_path_widget.mousePressEvent = self.open_output_folder

        second_line = QHBoxLayout()
        second_line.addWidget(self.res_output_path_widget)

        result_show_layout = QVBoxLayout()
        result_show_layout.addLayout(first_line)
        result_show_layout.addLayout(second_line)

        # Add layouts to main layout
        main_layout.addStretch()
        main_layout.addLayout(model_input_layout)
        main_layout.addStretch()
        main_layout.addWidget(line1)
        main_layout.addLayout(result_show_layout)
        main_layout.addStretch()
        self.setLayout(main_layout)
        # Initialize model and device
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def open_output_folder(self, event):
        output_path = self.res_output_path_widget.getText()
        if os.path.exists(output_path):
            folder_path = os.path.dirname(output_path)
            os.startfile(folder_path)
        else:
            QMessageBox.critical(self, "错误", "输出文件路径不存在")

    def load_model(self):
        config_file = "configs/pretrain.yml"
        checkpoint = self.ckp_selector.get_selected_file()
        if not config_file or not checkpoint:
            QMessageBox.critical(self, "错误", "请提供模型文件路径")
            return

        self.model = torch.load(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        # Warmup the model with a dummy input to ensure everything is loaded correctly
        dummy_input = torch.randn(1, 1, 16000).to(
            self.device
        )  # Assuming the model expects 1-second audio with 16kHz sampling rate
        with torch.no_grad():
            self.model.separate(dummy_input)

        QMessageBox.information(self, "提示", "模型加载成功")

    def audio_enhance(self):
        audio_input_path = self.audio_file_selector.get_selected_file()
        output_folder_path = self.output_folder_selector.get_selected_folder()
        if not self.model:
            QMessageBox.critical(self, "错误", "请先加载模型")
            return

        if not audio_input_path:
            QMessageBox.critical(self, "错误", "请先选择音频")
            return

        if not output_folder_path:
            QMessageBox.critical(self, "错误", "请先选择输出文件夹")
            return

        start_time = time.time()
        # enhance audio
        self.model.separate(
            audio_input_path,
            output_dir=output_folder_path,
            resample=True,
        )
        end_time = time.time()
        delta_time = end_time - start_time

        audio_output_path = os.path.join(
            output_folder_path,
            os.path.basename(audio_input_path).replace(".wav", "_est1.wav"),
        )

        audio_input, _ = torchaudio.load(audio_input_path)
        audio_input = audio_input.unsqueeze(0)
        audio_output, _ = torchaudio.load(audio_output_path)
        audio_output = audio_output.unsqueeze(0)

        # Ensure the audio tensors are on the same device as the model
        audio_input = audio_input.to(self.device)
        audio_output = audio_output.to(self.device)
        loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"), pit_from="pw_mtx").to(
            self.device
        )
        loss = loss_func(audio_output, audio_input)
        # show si-snr
        self.result_sisnr_widget.setText(f"{-1* loss.item():.2f} dB")
        # show time cost
        self.result_time_widget.setText(f"{delta_time*1000:.2f} ms")
        # show output path
        self.res_output_path_widget.setText(audio_output_path)
        QMessageBox.information(self, "提示", "降噪完毕！")


if __name__ == "__main__":
    os.chdir(r"C:\Users\june\Workspace\Asteroid\code")
    app = QApplication(sys.argv)
    main_window = InferenceInterface()
    main_window.show()
    sys.exit(app.exec_())
