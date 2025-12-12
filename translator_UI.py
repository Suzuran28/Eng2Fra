import sys
import re
import eng2fra
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QTextEdit, QLabel, 
    QStatusBar, QMessageBox, QGridLayout, QGroupBox
)
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QKeyEvent

class TranslatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('英法翻译器 (GRU模型)')
        self.setGeometry(300, 300, 500, 400)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 创建输入组
        input_group = QGroupBox("英文输入")
        input_layout = QVBoxLayout()
        
        self.input_field = QTextEdit()
        self.input_field.setMaximumHeight(100)
        self.input_field.setPlaceholderText("请输入英文文本，仅支持字母和!.?符号，最多10个单词...")
        self.input_field.installEventFilter(self)  # 安装事件过滤器以监控输入
        
        # 添加字符计数标签
        self.char_count_label = QLabel("单词数: 0/10")
        self.char_count_label.setAlignment(Qt.AlignRight)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.char_count_label)
        input_group.setLayout(input_layout)
        
        # 创建按钮组
        button_layout = QHBoxLayout()
        self.translate_button = QPushButton('翻译')
        self.translate_button.clicked.connect(self.translate_text)
        self.translate_button.setEnabled(False)  # 初始禁用翻译按钮
        self.clear_button = QPushButton('清空')
        self.clear_button.clicked.connect(self.clear_all)
        
        button_layout.addWidget(self.translate_button)
        button_layout.addWidget(self.clear_button)
        
        # 创建输出组
        output_group = QGroupBox("法文翻译结果")
        output_layout = QVBoxLayout()
        
        self.output_field = QTextEdit()
        self.output_field.setMaximumHeight(100)
        self.output_field.setReadOnly(True)
        self.output_field.setPlaceholderText("翻译结果将显示在这里...")
        
        output_layout.addWidget(self.output_field)
        output_group.setLayout(output_layout)
        
        # 示例文本按钮区域
        examples_group = QGroupBox("示例文本")
        examples_layout = QGridLayout()
        
        example_texts = [
            "Really?", "How are you?", "Stop!",
            "Thank you.", "See you later.", "I love you."
        ]
        
        for i, text in enumerate(example_texts):
            row = i // 3
            col = i % 3
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked, t=text: self.load_example(t))
            examples_layout.addWidget(btn, row, col)
        
        examples_group.setLayout(examples_layout)
        
        # 添加所有组件到主布局
        main_layout.addWidget(input_group)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(output_group)
        main_layout.addWidget(examples_group)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('准备就绪')
        
        # 连接文本变化信号
        self.input_field.textChanged.connect(self.on_text_changed)
        
    def eventFilter(self, obj, event):
        """事件过滤器，用于限制输入内容"""
        if obj == self.input_field and event.type() == QEvent.KeyPress:
            if isinstance(event, QKeyEvent):
                # 允许的按键: 字母、空格、退格、删除、Tab、Enter等控制键
                key = event.key()
                text = event.text()
                
                # 允许控制键
                if key in [Qt.Key_Backspace, Qt.Key_Delete, Qt.Key_Left, Qt.Key_Right,
                          Qt.Key_Up, Qt.Key_Down, Qt.Key_Home, Qt.Key_End, Qt.Key_Tab,
                          Qt.Key_Enter, Qt.Key_Return]:
                    return False  # 不拦截
                
                # 检查输入字符是否符合要求（字母、空格、标点）
                if text and not re.match(r'^[a-zA-Z\s\.\!\?]$', text):
                    return True  # 拦截不符合要求的输入
                
                # 检查单词数量是否超过限制
                current_text = self.input_field.toPlainText()
                words = current_text.split()
                if len(words) >= 10 and text and not text.isspace() and key not in [Qt.Key_Backspace, Qt.Key_Delete]:
                    # 如果已经有10个单词且不是删除操作，则阻止输入
                    if len(current_text.strip().split()) >= 10 and not text.isspace():
                        return True  # 拦截
                    
        return False  # 不拦截
        
    def on_text_changed(self):
        """当输入文本改变时触发"""
        text = self.input_field.toPlainText()
        
        # 过滤文本，只保留允许的字符
        filtered_text = re.sub(r'[^a-zA-Z\s\.\!\?]', '', text)
        
        # 限制单词数量为10个
        words = filtered_text.split()
        if len(words) > 10:
            filtered_text = ' '.join(words[:10])
            cursor_pos = self.input_field.textCursor().position()
            self.input_field.setPlainText(filtered_text)
            # 恢复光标位置
            cursor = self.input_field.textCursor()
            cursor.setPosition(min(cursor_pos, len(filtered_text)))
            self.input_field.setTextCursor(cursor)
        
        # 更新单词计数
        word_count = len(words) if len(words) <= 10 else 10
        self.char_count_label.setText(f"单词数: {word_count}/10")
        
        # 控制翻译按钮状态
        self.translate_button.setEnabled(len(filtered_text.strip()) > 0)
        
        # 如果文本被过滤，更新状态栏提示
        if text != filtered_text:
            self.status_bar.showMessage('注意：仅支持字母和!.?符号')
        
    def load_example(self, text):
        """加载示例文本"""
        self.input_field.setPlainText(text)
        self.status_bar.showMessage(f'已加载示例: "{text}"')
        
    def translate_text(self):
        """执行翻译操作"""
        input_text = self.input_field.toPlainText().strip()
        
        if not input_text:
            QMessageBox.warning(self, "输入错误", "请输入要翻译的英文文本！")
            return
            
        try:
            self.status_bar.showMessage('正在翻译...')
            QApplication.processEvents()  # 更新UI
            
            # 使用您提供的接口进行翻译
            result = eng2fra.use_seq2seq(eng2fra.normalizeEng(input_text))
            
            self.output_field.setPlainText(result)
            self.status_bar.showMessage('翻译完成')
        except Exception as e:
            QMessageBox.critical(self, "翻译错误", f"翻译过程中出现错误: {str(e)}")
            self.status_bar.showMessage('翻译失败')
            
    def clear_all(self):
        """清空所有输入输出"""
        self.input_field.clear()
        self.output_field.clear()
        self.char_count_label.setText("单词数: 0/10")
        self.status_bar.showMessage('已清空')


def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    window = TranslatorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()