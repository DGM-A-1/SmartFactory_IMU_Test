import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.font_manager as fm
from matplotlib.patches import Circle
import threading
import websocket
import json
from datetime import datetime, timedelta
import numpy as np
import platform
import sqlite3
import uuid
import os

class IMUGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🏭 Smart Factory IMU Monitoring System")
        self.root.configure(bg='#1a1a2e')  # 다크 네이비 배경
        
        self.fullscreen = False
        self.data_records = []
        self.pipeline = None
        self.streaming = False
        self.auto_mode = False
        self.collection_start_time = None
        self.threshold = 3.3
        self.session_id = None
        self.predictions_data = {}
        
        # 스레드 안전성을 위한 Lock 추가
        self.data_lock = threading.Lock()
        
        # 최대 레코드 수 제한
        self.MAX_RECORDS = 10000
        
        # WebSocket 연결 상태
        self.ws_connected = False
        self.connection_timeout = 5
        
        # 스마트 팩토리 색상 테마
        self.colors = {
            'bg_dark': '#1a1a2e',
            'bg_medium': '#16213e',
            'bg_light': '#0f3460',
            'accent': '#e94560',
            'success': '#00d25b',
            'warning': '#ffab00',
            'danger': '#fc424a',
            'info': '#00b8d9',
            'text_primary': '#ffffff',
            'text_secondary': '#8f9bb3',
            'grid': '#2a2d3a'
        }

        # 데이터베이스 초기화
        self.init_database()

        # 한글 폰트 설정
        self.setup_korean_font()
        
        # 다크 테마 스타일 설정
        plt.style.use('dark_background')
        
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))

        # 메인 레이아웃 구성
        self.setup_main_layout()

    def init_database(self):
        """데이터베이스 초기화"""
        self.db_path = "imu_analysis.db"
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS imu_raw_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    sensor_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    roll REAL NOT NULL,
                    pitch REAL NOT NULL,
                    yaw REAL NOT NULL,
                    x_del_ang REAL NOT NULL,
                    y_del_ang REAL NOT NULL,
                    z_del_ang REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS diagnosis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    sensor_id INTEGER NOT NULL,
                    measurement_date DATE NOT NULL,
                    measurement_time TIME NOT NULL,
                    data_collection_duration REAL NOT NULL,
                    predicted_roll_drift REAL,
                    predicted_pitch_drift REAL,
                    predicted_yaw_drift REAL,
                    max_drift_axis TEXT,
                    max_drift_value REAL,
                    max_drift_signed REAL,
                    is_faulty BOOLEAN NOT NULL,
                    fault_threshold REAL NOT NULL,
                    diagnosis_status TEXT NOT NULL,
                    model_version TEXT,
                    data_quality_score REAL,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS measurement_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    total_duration REAL,
                    sensor_count INTEGER,
                    total_data_points INTEGER,
                    session_type TEXT,
                    operator_name TEXT,
                    facility_location TEXT,
                    equipment_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            print("✅ 데이터베이스 초기화 완료")
            
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"❌ 데이터베이스 초기화 오류: {e}")
            messagebox.showerror("데이터베이스 오류", f"데이터베이스 초기화 실패:\n{e}")
        finally:
            if conn:
                conn.close()

    def setup_korean_font(self):
        """한글 폰트 설정"""
        system = platform.system()
        
        if system == 'Windows':
            font_candidates = ['Malgun Gothic', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        elif system == 'Darwin':
            font_candidates = ['AppleGothic', 'Arial Unicode MS', 'DejaVu Sans']
        else:
            font_candidates = ['DejaVu Sans', 'Liberation Sans', 'Noto Sans CJK KR']
        
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        korean_font = None
        
        for font in font_candidates:
            if font in available_fonts:
                korean_font = font
                break
        
        if korean_font:
            plt.rcParams['font.family'] = korean_font
            self.font_family = korean_font
            print(f"✅ 한글 폰트 설정: {korean_font}")
        else:
            plt.rcParams['axes.unicode_minus'] = False
            self.font_family = 'DejaVu Sans'
            print("⚠️ 한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")

    def setup_main_layout(self):
        """메인 레이아웃 구성"""
        # 상단 헤더
        self.setup_header()
        
        # 중앙 컨테이너
        main_container = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_container.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 왼쪽 패널 (상태 및 컨트롤)
        left_panel = tk.Frame(main_container, bg=self.colors['bg_medium'], width=350)
        left_panel.pack(side='left', fill='y', padx=(0, 5))
        left_panel.pack_propagate(False)
        
        self.setup_status_cards(left_panel)
        self.setup_control_buttons(left_panel)
        
        # 오른쪽 패널 (그래프)
        right_panel = tk.Frame(main_container, bg=self.colors['bg_dark'])
        right_panel.pack(side='right', fill='both', expand=True)
        
        self.setup_plots(right_panel)
        
        # 하단 상태바
        self.setup_statusbar()

    def setup_header(self):
        """상단 헤더 구성"""
        header = tk.Frame(self.root, bg=self.colors['bg_medium'], height=80)
        header.pack(fill='x', padx=0, pady=0)
        header.pack_propagate(False)
        
        # 로고/타이틀 영역
        title_frame = tk.Frame(header, bg=self.colors['bg_medium'])
        title_frame.pack(side='left', padx=20, pady=10)
        
        tk.Label(title_frame, 
                text="🏭 SMART FACTORY",
                font=(self.font_family, 24, 'bold'),
                bg=self.colors['bg_medium'],
                fg=self.colors['text_primary']).pack(anchor='w')
        
        tk.Label(title_frame,
                text="IMU Real-Time Monitoring & Analysis System",
                font=(self.font_family, 12),
                bg=self.colors['bg_medium'],
                fg=self.colors['text_secondary']).pack(anchor='w')
        
        # 실시간 시계
        self.time_label = tk.Label(header,
                                  text="",
                                  font=(self.font_family, 14),
                                  bg=self.colors['bg_medium'],
                                  fg=self.colors['info'])
        self.time_label.pack(side='right', padx=20)
        self.update_time()

    def update_time(self):
        """실시간 시계 업데이트"""
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=f"🕒 {time_str}")
        self.root.after(1000, self.update_time)

    def setup_status_cards(self, parent):
        """상태 카드 구성"""
        # 연결 상태 카드
        conn_card = self.create_card(parent, "CONNECTION STATUS")
        self.conn_indicator = self.create_status_indicator(conn_card, "WebSocket", "OFFLINE")
        
        # 모델 상태 카드
        model_card = self.create_card(parent, "AI MODEL STATUS")
        self.model_indicator = self.create_status_indicator(model_card, "ML Pipeline", "NOT LOADED")
        
        # 데이터 수집 카드
        data_card = self.create_card(parent, "DATA COLLECTION")
        
        self.data_count_label = tk.Label(data_card,
                                        text="Records: 0",
                                        font=(self.font_family, 11),
                                        bg=self.colors['bg_light'],
                                        fg=self.colors['text_primary'])
        self.data_count_label.pack(pady=5)
        
        self.session_label = tk.Label(data_card,
                                     text="Session: N/A",
                                     font=(self.font_family, 9),
                                     bg=self.colors['bg_light'],
                                     fg=self.colors['text_secondary'])
        self.session_label.pack()
        
        # 측정 상태 카드
        measure_card = self.create_card(parent, "MEASUREMENT")
        
        self.measure_status = tk.Label(measure_card,
                                      text="⏸ STANDBY",
                                      font=(self.font_family, 14, 'bold'),
                                      bg=self.colors['bg_light'],
                                      fg=self.colors['text_secondary'])
        self.measure_status.pack(pady=10)
        
        self.countdown_label = tk.Label(measure_card,
                                       text="",
                                       font=(self.font_family, 20, 'bold'),
                                       bg=self.colors['bg_light'],
                                       fg=self.colors['warning'])
        self.countdown_label.pack()
        
        # 임계값 설정 카드
        threshold_card = self.create_card(parent, "FAULT THRESHOLD")
        
        threshold_frame = tk.Frame(threshold_card, bg=self.colors['bg_light'])
        threshold_frame.pack(pady=10)
        
        tk.Label(threshold_frame,
                text="Drift Limit:",
                font=(self.font_family, 10),
                bg=self.colors['bg_light'],
                fg=self.colors['text_secondary']).pack(side='left', padx=5)
        
        self.threshold_display = tk.Label(threshold_frame,
                                         text=f"{self.threshold}°",
                                         font=(self.font_family, 16, 'bold'),
                                         bg=self.colors['bg_light'],
                                         fg=self.colors['accent'])
        self.threshold_display.pack(side='left')

    def create_card(self, parent, title):
        """상태 카드 생성"""
        card = tk.Frame(parent, bg=self.colors['bg_light'], relief='flat', bd=0)
        card.pack(fill='x', padx=10, pady=5)
        
        # 카드 헤더
        header = tk.Frame(card, bg=self.colors['accent'], height=3)
        header.pack(fill='x')
        
        tk.Label(card,
                text=title,
                font=(self.font_family, 10, 'bold'),
                bg=self.colors['bg_light'],
                fg=self.colors['text_secondary']).pack(pady=(10, 5))
        
        return card

    def create_status_indicator(self, parent, label, status):
        """상태 인디케이터 생성"""
        frame = tk.Frame(parent, bg=self.colors['bg_light'])
        frame.pack(pady=10)
        
        # 상태 LED
        self.led_canvas = tk.Canvas(frame, width=20, height=20, 
                                   bg=self.colors['bg_light'], highlightthickness=0)
        self.led_canvas.pack(side='left', padx=5)
        
        led = self.led_canvas.create_oval(2, 2, 18, 18, 
                                         fill=self.colors['danger'], 
                                         outline=self.colors['danger'])
        
        # 상태 텍스트
        status_label = tk.Label(frame,
                              text=f"{label}: {status}",
                              font=(self.font_family, 10),
                              bg=self.colors['bg_light'],
                              fg=self.colors['text_primary'])
        status_label.pack(side='left')
        
        return {'canvas': self.led_canvas, 'led': led, 'label': status_label}

    def setup_control_buttons(self, parent):
        """컨트롤 버튼 구성"""
        btn_frame = tk.Frame(parent, bg=self.colors['bg_medium'])
        btn_frame.pack(fill='x', padx=10, pady=20)
        
        button_config = {
            'font': (self.font_family, 11, 'bold'),
            'relief': 'flat',
            'cursor': 'hand2',
            'activebackground': self.colors['accent']
        }
        
        # 자동 측정 버튼
        self.auto_btn = tk.Button(btn_frame,
                                 text="🚀 START AUTO MEASUREMENT",
                                 command=self.start_auto_collection,
                                 bg=self.colors['success'],
                                 fg='white',
                                 height=2,
                                 **button_config)
        self.auto_btn.pack(fill='x', pady=3)
        
        # ML 모델 로드 버튼
        tk.Button(btn_frame,
                 text="🤖 LOAD AI MODEL",
                 command=self.load_model,
                 bg=self.colors['info'],
                 fg='white',
                 height=2,
                 **button_config).pack(fill='x', pady=3)
        
        # 데이터 저장 버튼
        tk.Button(btn_frame,
                 text="💾 SAVE TO EXCEL",
                 command=self.save_data,
                 bg=self.colors['warning'],
                 fg='white',
                 **button_config).pack(fill='x', pady=3)
        
        # DB 저장 버튼
        tk.Button(btn_frame,
                 text="🗄️ SAVE TO DATABASE",
                 command=self.save_to_database,
                 bg='#7c3aed',
                 fg='white',
                 **button_config).pack(fill='x', pady=3)
        
        # 데이터 초기화 버튼
        tk.Button(btn_frame,
                 text="🔄 RESET DATA",
                 command=self.clear_data,
                 bg=self.colors['grid'],
                 fg='white',
                 **button_config).pack(fill='x', pady=3)

    def setup_plots(self, parent):
        """그래프 설정"""
        # 그래프 컨테이너
        plot_frame = tk.Frame(parent, bg=self.colors['bg_medium'])
        plot_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 그래프 생성
        self.fig, axs = plt.subplots(4, 2, figsize=(14, 10), facecolor=self.colors['bg_dark'])
        self.axes = axs.flatten()
        
        for i, ax in enumerate(self.axes):
            ax.set_facecolor(self.colors['bg_medium'])
            ax.set_title(f"🔧 SENSOR {i}", fontsize=12, fontweight='bold', 
                        color=self.colors['text_primary'], fontfamily=self.font_family)
            ax.set_ylabel("Angle (°)", fontsize=10, color=self.colors['text_secondary'], 
                         fontfamily=self.font_family)
            ax.grid(True, alpha=0.2, color=self.colors['grid'])
            ax.tick_params(colors=self.colors['text_secondary'])
            
            # 축 색상 설정
            for spine in ax.spines.values():
                spine.set_edgecolor(self.colors['grid'])
        
        # X축 레이블은 하단 두 개만
        self.axes[6].set_xlabel("Time", fontsize=10, color=self.colors['text_secondary'], 
                                fontfamily=self.font_family)
        self.axes[7].set_xlabel("Time", fontsize=10, color=self.colors['text_secondary'], 
                                fontfamily=self.font_family)
        
        plt.tight_layout(pad=2.5)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def setup_statusbar(self):
        """하단 상태바 구성"""
        statusbar = tk.Frame(self.root, bg=self.colors['bg_medium'], height=30)
        statusbar.pack(side='bottom', fill='x')
        statusbar.pack_propagate(False)
        
        # 왼쪽 상태 메시지
        self.status_message = tk.Label(statusbar,
                                      text="✅ System Ready",
                                      font=(self.font_family, 10),
                                      bg=self.colors['bg_medium'],
                                      fg=self.colors['text_secondary'])
        self.status_message.pack(side='left', padx=20)
        
        # 오른쪽 정보
        info_frame = tk.Frame(statusbar, bg=self.colors['bg_medium'])
        info_frame.pack(side='right', padx=20)
        
        tk.Label(info_frame,
                text="v2.0 | Smart Factory Edition",
                font=(self.font_family, 9),
                bg=self.colors['bg_medium'],
                fg=self.colors['text_secondary']).pack(side='right')

    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def update_status(self, message, status_type='info'):
        """상태 업데이트"""
        colors = {
            'info': self.colors['info'],
            'success': self.colors['success'],
            'warning': self.colors['warning'],
            'danger': self.colors['danger']
        }
        
        icons = {
            'info': 'ℹ️',
            'success': '✅',
            'warning': '⚠️',
            'danger': '❌'
        }
        
        self.status_message.config(
            text=f"{icons.get(status_type, '')} {message}",
            fg=colors.get(status_type, self.colors['text_secondary'])
        )
        self.root.update()

    def update_connection_status(self, connected):
        """연결 상태 업데이트"""
        if connected:
            self.conn_indicator['canvas'].itemconfig(
                self.conn_indicator['led'], 
                fill=self.colors['success'],
                outline=self.colors['success']
            )
            self.conn_indicator['label'].config(text="WebSocket: ONLINE")
        else:
            self.conn_indicator['canvas'].itemconfig(
                self.conn_indicator['led'],
                fill=self.colors['danger'],
                outline=self.colors['danger']
            )
            self.conn_indicator['label'].config(text="WebSocket: OFFLINE")

    def update_model_status(self, loaded):
        """모델 상태 업데이트"""
        if loaded:
            self.model_indicator['canvas'].itemconfig(
                self.model_indicator['led'],
                fill=self.colors['success'],
                outline=self.colors['success']
            )
            self.model_indicator['label'].config(text="ML Pipeline: READY")
        else:
            self.model_indicator['canvas'].itemconfig(
                self.model_indicator['led'],
                fill=self.colors['danger'],
                outline=self.colors['danger']
            )
            self.model_indicator['label'].config(text="ML Pipeline: NOT LOADED")

    def update_data_count(self):
        """데이터 카운트 업데이트"""
        with self.data_lock:
            count = len(self.data_records)
        self.data_count_label.config(text=f"Records: {count:,}")
        
        if self.session_id:
            short_id = self.session_id[:8]
            self.session_label.config(text=f"Session: {short_id}...")

    def wait_for_connection(self, callback):
        """WebSocket 연결을 기다리고 연결 상태를 확인"""
        start_time = datetime.now()
        
        def check_connection():
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if self.ws_connected:
                callback(True)
            elif elapsed >= self.connection_timeout:
                callback(False)
            else:
                self.root.after(100, check_connection)
        
        check_connection()

    def start_auto_collection(self):
        """자동 측정 시작 - 5초 후 자동 종료 및 예측"""
        if self.pipeline is None:
            messagebox.showerror("모델 오류", "먼저 AI 모델을 로드해주세요!")
            return
            
        self.auto_mode = True
        with self.data_lock:
            self.data_records = []
        self.predictions_data = {}
        self.collection_start_time = datetime.now()
        self.session_id = str(uuid.uuid4())
        
        self.save_session_info()
        
        self.measure_status.config(text="🔄 CONNECTING...", fg=self.colors['warning'])
        self.update_status("WebSocket 연결 중...", 'warning')
        self.start_stream()
        
        def on_connection_result(connected):
            if connected:
                self.measure_status.config(text="📊 COLLECTING DATA", fg=self.colors['success'])
                self.update_status("자동 측정 진행 중 (5초)", 'success')
                self.start_countdown(5)
            else:
                self.auto_mode = False
                self.stop_stream()
                self.measure_status.config(text="❌ CONNECTION FAILED", fg=self.colors['danger'])
                self.update_status("WebSocket 서버 연결 실패", 'danger')
                messagebox.showerror("연결 오류", 
                    "WebSocket 서버에 연결할 수 없습니다.\n"
                    "서버 상태를 확인해주세요.")
        
        self.wait_for_connection(on_connection_result)

    def save_session_info(self):
        """측정 세션 정보를 데이터베이스에 저장"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("BEGIN TRANSACTION")
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO measurement_sessions 
                (session_id, start_time, session_type, operator_name, facility_location, equipment_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                self.session_id,
                self.collection_start_time.isoformat(),
                "자동",
                "운영자",
                "시설위치",
                "IMU-001"
            ))
            
            conn.commit()
            self.update_data_count()
            
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"세션 정보 저장 오류: {e}")
        finally:
            if conn:
                conn.close()

    def start_countdown(self, seconds_left):
        if seconds_left > 0 and self.auto_mode:
            self.countdown_label.config(text=f"{seconds_left}")
            self.root.after(1000, lambda: self.start_countdown(seconds_left - 1))
        elif self.auto_mode:
            self.countdown_label.config(text="")
            self.measure_status.config(text="🔍 ANALYZING...", fg=self.colors['info'])
            self.stop_stream()
            self.root.after(500, self.predict)

    def clear_data(self):
        """데이터 초기화"""
        with self.data_lock:
            self.data_records = []
        self.predictions_data = {}
        self.update_data_count()
        
        for ax in self.axes:
            ax.cla()
            ax.set_facecolor(self.colors['bg_medium'])
            ax.set_title(f"🔧 SENSOR {self.axes.tolist().index(ax)}", fontsize=12, fontweight='bold',
                        color=self.colors['text_primary'], fontfamily=self.font_family)
            ax.set_ylabel("Angle (°)", fontsize=10, color=self.colors['text_secondary'],
                         fontfamily=self.font_family)
            ax.grid(True, alpha=0.2, color=self.colors['grid'])
            ax.tick_params(colors=self.colors['text_secondary'])
            
            for spine in ax.spines.values():
                spine.set_edgecolor(self.colors['grid'])
        
        self.canvas.draw()
        self.measure_status.config(text="⏸ STANDBY", fg=self.colors['text_secondary'])
        self.update_status("데이터 초기화 완료", 'success')

    def on_message(self, ws, message):
        try:
            msg = json.loads(message)
            ts = datetime.now()
            
            with self.data_lock:
                if 'sensors' in msg:
                    for sensor in msg['sensors']:
                        rec = sensor.copy()
                        rec['SN'] = rec.get('id')
                        rec['timestamp'] = ts
                        self.data_records.append(rec)
                else:
                    msg['timestamp'] = ts
                    self.data_records.append(msg)
                
                if len(self.data_records) > self.MAX_RECORDS:
                    self.data_records = self.data_records[-self.MAX_RECORDS:]
            
            self.update_data_count()
                    
        except Exception as e:
            print("메시지 파싱 오류:", e)

    def on_error(self, ws, error):
        print("WebSocket 오류:", error)
        self.ws_connected = False
        self.update_connection_status(False)
        self.update_status("연결 오류 발생", 'danger')

    def on_close(self, ws, close_status, close_msg):
        print("WebSocket 연결 종료")
        self.ws_connected = False
        self.update_connection_status(False)
        if self.streaming:
            self.update_status("연결이 끊어졌습니다", 'warning')

    def on_open(self, ws):
        print("WebSocket 연결 성공")
        self.ws_connected = True
        self.update_connection_status(True)
        self.update_status("데이터 수집 중", 'success')

    def start_stream(self):
        if self.streaming:
            return
        
        self.ws_connected = False
        ws_url = "ws://10.200.246.81:81"
        
        try:
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            self.wst = threading.Thread(target=self.ws.run_forever)
            self.wst.daemon = True
            self.streaming = True
            self.wst.start()
            self.root.after(100, self.update_plot)
        except Exception as e:
            print(f"WebSocket 시작 오류: {e}")
            self.streaming = False
            self.ws_connected = False

    def stop_stream(self):
        if not self.streaming:
            return
        self.streaming = False
        self.ws_connected = False
        self.update_connection_status(False)
        try:
            self.ws.close()
        except:
            pass
        
        if self.auto_mode:
            self.countdown_label.config(text="")
            self.auto_mode = False
        else:
            self.update_status("데이터 수집 중지", 'info')

    def update_plot(self):
        if not self.streaming:
            return
        
        try:
            with self.data_lock:
                if self.data_records:
                    df = pd.DataFrame(self.data_records.copy())
            
            if 'df' in locals() and not df.empty:
                if 'timestamp' not in df.columns:
                    self.root.after(100, self.update_plot)
                    return
                    
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                for ax in self.axes:
                    ax.cla()
                    ax.set_facecolor(self.colors['bg_medium'])
                    ax.grid(True, alpha=0.2, color=self.colors['grid'])
                    ax.tick_params(colors=self.colors['text_secondary'])
                    
                    for spine in ax.spines.values():
                        spine.set_edgecolor(self.colors['grid'])
                    
                for sn in range(8):
                    ax = self.axes[sn]
                    ax.set_title(f"🔧 SENSOR {sn}", fontsize=12, fontweight='bold',
                               color=self.colors['text_primary'], fontfamily=self.font_family)
                    
                    if 'SN' in df.columns:
                        sub = df[df['SN'] == sn]
                        if not sub.empty and all(col in sub.columns for col in ['ROLL', 'PITCH', 'YAW']):
                            ax.plot(sub['timestamp'], sub['ROLL'], '#ff6b6b', linewidth=2, label='Roll', alpha=0.9)
                            ax.plot(sub['timestamp'], sub['PITCH'], '#4ecdc4', linewidth=2, label='Pitch', alpha=0.9)
                            ax.plot(sub['timestamp'], sub['YAW'], '#45b7d1', linewidth=2, label='Yaw', alpha=0.9)
                            ax.legend(loc='upper right', fontsize=9, framealpha=0.9,
                                    facecolor=self.colors['bg_dark'], edgecolor='none',
                                    prop={'family': self.font_family})
                    
                    ax.set_ylabel("Angle (°)", fontsize=10, color=self.colors['text_secondary'],
                                 fontfamily=self.font_family)
                    if sn >= 6:
                        ax.set_xlabel("Time", fontsize=10, color=self.colors['text_secondary'],
                                     fontfamily=self.font_family)
                
                plt.tight_layout(pad=2.0)
                self.canvas.draw()
                
        except Exception as e:
            print(f"플롯 업데이트 오류: {e}")
        
        self.root.after(100, self.update_plot)

    def save_data(self):
        with self.data_lock:
            if not self.data_records:
                messagebox.showwarning("경고", "저장할 데이터가 없습니다")
                return
            
            df = pd.DataFrame(self.data_records.copy())
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel 파일","*.xlsx")]
        )
        if file_path:
            try:
                df.to_excel(file_path, index=False)
                self.update_status(f"데이터 저장 완료: {os.path.basename(file_path)}", 'success')
                messagebox.showinfo("성공", f"데이터가 저장되었습니다:\n{file_path}")
            except Exception as e:
                self.update_status("파일 저장 실패", 'danger')
                messagebox.showerror("오류", f"파일 저장 실패:\n{e}")

    def save_to_database(self):
        """수집된 데이터와 예측 결과를 데이터베이스에 저장"""
        with self.data_lock:
            if not self.data_records:
                messagebox.showwarning("경고", "저장할 데이터가 없습니다")
                return
            
            data_to_save = self.data_records.copy()
        
        if not self.predictions_data:
            messagebox.showwarning("경고", "예측 결과가 없습니다. 먼저 자동 측정을 실행해주세요.")
            return
        
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("BEGIN TRANSACTION")
            cursor = conn.cursor()
            
            df = pd.DataFrame(data_to_save)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            raw_data_count = 0
            for _, row in df.iterrows():
                if all(col in row for col in ['SN', 'ROLL', 'PITCH', 'YAW', 'X_DEL_ANG', 'Y_DEL_ANG', 'Z_DEL_ANG']):
                    cursor.execute('''
                        INSERT INTO imu_raw_data 
                        (session_id, sensor_id, timestamp, roll, pitch, yaw, x_del_ang, y_del_ang, z_del_ang)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        self.session_id,
                        int(row['SN']),
                        row['timestamp'].isoformat(),
                        float(row['ROLL']),
                        float(row['PITCH']),
                        float(row['YAW']),
                        float(row['X_DEL_ANG']),
                        float(row['Y_DEL_ANG']),
                        float(row['Z_DEL_ANG'])
                    ))
                    raw_data_count += 1
            
            measurement_time = self.collection_start_time
            end_time = datetime.now()
            duration = (end_time - self.collection_start_time).total_seconds()
            
            diagnosis_count = 0
            for sensor_id, pred_info in self.predictions_data.items():
                cursor.execute('''
                    INSERT INTO diagnosis_results 
                    (session_id, sensor_id, measurement_date, measurement_time, data_collection_duration,
                     predicted_roll_drift, predicted_pitch_drift, predicted_yaw_drift,
                     max_drift_axis, max_drift_value, max_drift_signed,
                     is_faulty, fault_threshold, diagnosis_status, model_version, data_quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.session_id,
                    sensor_id,
                    measurement_time.date().isoformat(),
                    measurement_time.time().isoformat(),
                    duration,
                    pred_info.get('roll_drift'),
                    pred_info.get('pitch_drift'),
                    pred_info.get('yaw_drift'),
                    pred_info.get('max_drift_axis'),
                    pred_info.get('max_drift_value'),
                    pred_info.get('max_drift_signed'),
                    pred_info.get('is_faulty', False),
                    self.threshold,
                    pred_info.get('status', '정상'),
                    "v1.0",
                    1.0
                ))
                diagnosis_count += 1
            
            active_sensors = len(set(df['SN'])) if 'SN' in df.columns else 0
            cursor.execute('''
                UPDATE measurement_sessions 
                SET end_time = ?, total_duration = ?, sensor_count = ?, total_data_points = ?
                WHERE session_id = ?
            ''', (
                end_time.isoformat(),
                duration,
                active_sensors,
                len(df),
                self.session_id
            ))
            
            conn.commit()
            
            self.update_status("데이터베이스 저장 완료", 'success')
            messagebox.showinfo("성공", 
                f"데이터베이스에 저장 완료!\n"
                f"- 원시 데이터: {raw_data_count:,}개\n"
                f"- 진단 결과: {diagnosis_count}개\n"
                f"- 세션 ID: {self.session_id[:8]}...")
            
        except Exception as e:
            if conn:
                conn.rollback()
            self.update_status("데이터베이스 저장 실패", 'danger')
            messagebox.showerror("오류", f"데이터베이스 저장 실패:\n{e}")
            print(f"DB 저장 오류: {e}")
        finally:
            if conn:
                conn.close()

    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="AI 모델 파일 선택",
            filetypes=[("Pickle 파일","*.pkl")]
        )
        if not file_path:
            return
        
        try:
            self.pipeline = joblib.load(file_path)
            self.update_model_status(True)
            self.update_status("AI 모델 로드 완료", 'success')
            messagebox.showinfo("성공", "AI 모델이 성공적으로 로드되었습니다")
        except Exception as e:
            self.update_model_status(False)
            self.update_status("모델 로드 실패", 'danger')
            messagebox.showerror("오류", f"모델 로드 실패:\n{e}")

    def predict(self):
        if self.pipeline is None:
            messagebox.showerror("오류", "AI 모델이 로드되지 않았습니다")
            return
        
        with self.data_lock:
            if not self.data_records:
                messagebox.showwarning("경고", "예측할 데이터가 없습니다")
                return
            
            data_for_prediction = self.data_records.copy()

        try:
            df = pd.DataFrame(data_for_prediction)
            
            required_cols = ['timestamp', 'SN', 'ROLL', 'PITCH', 'YAW', 'X_DEL_ANG', 'Y_DEL_ANG', 'Z_DEL_ANG']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                messagebox.showerror("오류", f"필수 데이터 컬럼이 없습니다: {missing_cols}")
                return
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            predictions = {}
            self.predictions_data = {}

            for sn in range(8):
                sub = df[df['SN'] == sn].sort_values('timestamp')
                if len(sub) < 2:
                    continue

                t0 = sub['timestamp'].iloc[0]
                start_time = t0 + timedelta(seconds=1)
                end_time = start_time + timedelta(seconds=4)
                window = sub[(sub['timestamp'] >= start_time) & (sub['timestamp'] <= end_time)]
                
                if len(window) < 2:
                    continue

                p5 = (-window['X_DEL_ANG']).mean()
                q5 = (-window['Z_DEL_ANG']).mean()
                r5 = ( window['Y_DEL_ANG']).mean()

                dt = (window['timestamp'].iloc[-1] - window['timestamp'].iloc[0]).total_seconds()
                if dt == 0:
                    continue
                    
                Rd5 = (window['ROLL'].iloc[-1] - window['ROLL'].iloc[0]) / dt
                Pd5 = (window['PITCH'].iloc[-1] - window['PITCH'].iloc[0]) / dt
                Yd5 = (window['YAW'].iloc[-1] - window['YAW'].iloc[0]) / dt

                R = window['ROLL'].values
                P = window['PITCH'].values
                p = -window['X_DEL_ANG'].values
                q = -window['Z_DEL_ANG'].values
                r = window['Y_DEL_ANG'].values

                R_rad = np.deg2rad(R)
                P_rad = np.deg2rad(P)
                
                cos_P = np.cos(P_rad)
                cos_P = np.where(np.abs(cos_P) < 1e-10, 1e-10 * np.sign(cos_P), cos_P)
                
                sin_R = np.sin(R_rad)
                cos_R = np.cos(R_rad)
                tan_P = np.tan(P_rad)
                tan_P = np.clip(tan_P, -100, 100)
                
                Rdot = p + (q * sin_R + r * cos_R) * tan_P
                Pdot = q * cos_R - r * sin_R
                Ydot = (q * sin_R + r * cos_R) / cos_P
                
                Rdot = np.nan_to_num(Rdot, nan=0.0, posinf=0.0, neginf=0.0)
                Pdot = np.nan_to_num(Pdot, nan=0.0, posinf=0.0, neginf=0.0)
                Ydot = np.nan_to_num(Ydot, nan=0.0, posinf=0.0, neginf=0.0)

                Rdot5 = Rdot.mean()
                Pdot5 = Pdot.mean()
                Ydot5 = Ydot.mean()

                X_feat = pd.DataFrame([[p5, q5, r5, Rd5, Pd5, Yd5, Rdot5, Pdot5, Ydot5]],
                                      columns=['p5','q5','r5','Rd5','Pd5','Yd5','Rdot5','Pdot5','Ydot5'])
                
                if X_feat.isnull().any().any():
                    print(f"센서 {sn}: 특성 계산 중 NaN 발생, 건너뜀")
                    continue
                
                try:
                    raw_pred = self.pipeline.predict(X_feat)
                    pred_vals = raw_pred[0] if hasattr(raw_pred[0], '__len__') else [raw_pred[0]]
                except Exception as e:
                    print(f"센서 {sn} 예측 오류: {e}")
                    continue

                predictions[sn] = pred_vals
                
                if len(pred_vals) == 3:
                    r_pred, p_pred, y_pred = pred_vals
                    drift_values = {'Roll': abs(r_pred), 'Pitch': abs(p_pred), 'Yaw': abs(y_pred)}
                    max_drift_axis = max(drift_values, key=drift_values.get)
                    max_drift_value = max(drift_values.values())
                    
                    if max_drift_axis == 'Roll':
                        max_drift_signed = r_pred
                    elif max_drift_axis == 'Pitch':
                        max_drift_signed = p_pred
                    else:
                        max_drift_signed = y_pred
                    
                    is_faulty = max_drift_value > self.threshold
                    status = "고장" if is_faulty else "정상"
                    
                    self.predictions_data[sn] = {
                        'roll_drift': float(r_pred),
                        'pitch_drift': float(p_pred),
                        'yaw_drift': float(y_pred),
                        'max_drift_axis': max_drift_axis,
                        'max_drift_value': float(max_drift_value),
                        'max_drift_signed': float(max_drift_signed),
                        'is_faulty': is_faulty,
                        'status': status
                    }
                else:
                    val = pred_vals[0]
                    is_faulty = abs(val) > self.threshold
                    status = "고장" if is_faulty else "정상"
                    
                    self.predictions_data[sn] = {
                        'roll_drift': None,
                        'pitch_drift': None,
                        'yaw_drift': None,
                        'max_drift_axis': 'Unknown',
                        'max_drift_value': float(abs(val)),
                        'max_drift_signed': float(val),
                        'is_faulty': is_faulty,
                        'status': status
                    }

            self.display_predictions(predictions)
            
            if self.auto_mode:
                self.measure_status.config(text="✅ ANALYSIS COMPLETE", fg=self.colors['success'])
                self.update_status("자동 분석 완료", 'success')
            else:
                messagebox.showinfo("완료", "예측이 완료되었습니다")
                
        except Exception as e:
            self.update_status("예측 중 오류 발생", 'danger')
            messagebox.showerror("오류", f"예측 중 오류 발생:\n{e}")
            print(f"예측 오류 상세: {e}")

    def display_predictions(self, predictions):
        """예측 결과를 그래프에 표시"""
        for sn, ax in enumerate(self.axes):
            pred = predictions.get(sn)
            
            for txt in ax.texts:
                txt.remove()
            
            if pred is None or len(pred) == 0:
                label = "DATA\nINSUFFICIENT"
                color = self.colors['text_secondary']
                bgcolor = self.colors['bg_dark']
                border_color = self.colors['grid']
            else:
                if len(pred) == 3:
                    r_pred, p_pred, y_pred = pred
                    drift_values = {'Roll': abs(r_pred), 'Pitch': abs(p_pred), 'Yaw': abs(y_pred)}
                    max_drift_axis = max(drift_values, key=drift_values.get)
                    max_drift_value = max(drift_values.values())
                    
                    if max_drift_axis == 'Roll':
                        max_drift_signed = r_pred
                    elif max_drift_axis == 'Pitch':
                        max_drift_signed = p_pred
                    else:
                        max_drift_signed = y_pred
                    
                    fail = max_drift_value > self.threshold
                    status = "⚠️ FAULT" if fail else "✅ NORMAL"
                    color = self.colors['text_primary']
                    bgcolor = self.colors['danger'] if fail else self.colors['success']
                    border_color = '#ff0000' if fail else '#00ff00'
                    
                    label = f"100s DRIFT PREDICTION\n{max_drift_axis}: {max_drift_signed:.2f}°\n{status}"
                else:
                    val = pred[0]
                    fail = abs(val) > self.threshold
                    status = "⚠️ FAULT" if fail else "✅ NORMAL"
                    color = self.colors['text_primary']
                    bgcolor = self.colors['danger'] if fail else self.colors['success']
                    border_color = '#ff0000' if fail else '#00ff00'
                    label = f"100s DRIFT\n{val:.2f}°\n{status}"
            
            ax.text(0.5, 0.95, label, transform=ax.transAxes,
                    ha='center', va='top', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=bgcolor, 
                             edgecolor=border_color, linewidth=2, alpha=0.9),
                    color=color, fontfamily=self.font_family)

        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    root.state('zoomed')
    app = IMUGUI(root)
    root.mainloop()