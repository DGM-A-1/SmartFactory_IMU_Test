# -*- coding: utf-8 -*-

# GUI.py
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
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
import requests  # ✅ API 호출용

class LoginDialog(simpledialog.Dialog):
    """이메일/비밀번호를 한 번에 입력받는 모달 다이얼로그"""
    def __init__(self, parent, title="로그인"):
        self.email = None
        self.password = None
        super().__init__(parent, title)

    def body(self, master):
        master.columnconfigure(1, weight=1)

        tk.Label(master, text="이메일:", font=("Arial", 11, "bold")).grid(row=0, column=0, sticky="w", padx=8, pady=(10, 4))
        self.email_var = tk.StringVar()
        self.email_entry = ttk.Entry(master, textvariable=self.email_var, width=32)
        self.email_entry.grid(row=0, column=1, sticky="we", padx=(0, 10), pady=(10, 4))

        tk.Label(master, text="비밀번호:", font=("Arial", 11, "bold")).grid(row=1, column=0, sticky="w", padx=8, pady=4)
        self.pw_var = tk.StringVar()
        self.pw_entry = ttk.Entry(master, textvariable=self.pw_var, show="*", width=32)
        self.pw_entry.grid(row=1, column=1, sticky="we", padx=(0, 10), pady=4)

        # 비밀번호 표시 토글
        self.show_var = tk.BooleanVar(value=False)
        self.show_chk = ttk.Checkbutton(master, text="비밀번호 표시", variable=self.show_var, command=self._toggle_pw)
        self.show_chk.grid(row=2, column=1, sticky="w", padx=(0, 10), pady=(0, 6))

        # 처음 포커스 위치
        return self.email_entry

    def _toggle_pw(self):
        self.pw_entry.configure(show="" if self.show_var.get() else "*")

    def validate(self):
        email = (self.email_var.get() or "").strip()
        pw = self.pw_var.get()
        if not email:
            messagebox.showwarning("입력 필요", "이메일을 입력하세요.")
            return False
        if pw is None or pw == "":
            messagebox.showwarning("입력 필요", "비밀번호를 입력하세요.")
            return False
        return True

    def apply(self):
        self.email = (self.email_var.get() or "").strip()
        self.password = self.pw_var.get()


class IMUGUI:
    # ----------------- 생성자 -----------------
    def __init__(self, root):
        self.root = root
        self.root.title("🏭 Smart Factory IMU Monitoring System")
        self.root.configure(bg='#f8f9fa')
        
        self.fullscreen = False
        self.data_records = []
        self.pipeline = None
        self.streaming = False
        self.auto_mode = False
        self.collection_start_time = None
        self.threshold = 3.3
        self.session_id = None
        self.predictions_data = {}
        
        self.data_lock = threading.Lock()
        self.MAX_RECORDS = 10000
        
        self.ws_connected = False
        self.connection_timeout = 10  # ⬅️ 타임아웃 살짝 여유

        # ✅ 카운트다운 시작 여부(스레드 안전한 UI 트리거용)
        self._countdown_started = False
        
        self.colors = {
            'bg_dark': '#f8f9fa',
            'bg_medium': '#ffffff',
            'bg_light': '#f1f3f5',
            'accent': '#e94560',
            'success': '#00c896',
            'warning': '#ff9800',
            'danger': '#dc3545',
            'info': '#17a2b8',
            'text_primary': '#212529',
            'text_secondary': '#6c757d',
            'grid': '#dee2e6'
        }

        # === Local DB 저장을 위한 입력값 ===
        # 예: sqlite:///./smartfactory.db  또는  C:/data/smartfactory.db
        self.db_url_var = tk.StringVar(value="sqlite:///./smartfactory.db")
        self.operator_name = "admin"     # 고정 표기(레거시)
        self.upload_box_no_var = tk.StringVar(value="")
        # Destination은 Room3으로 고정, Arrived는 업로드 시 항상 True로 저장합니다.

        # ✅ API 로그인/설정 상태
        self.api_base_var = tk.StringVar(value="http://127.0.0.1:8000")
        self.auth_token = None
        self.user_summary = None
        self.operator_name_var = tk.StringVar(value=self.operator_name)

        # 로컬(내부용) 분석 DB 초기화 (기존 유지)
        self.init_database()
        self.setup_korean_font()
        plt.style.use('default')
        
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        self.setup_main_layout()

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    # ----------------- 내부 분석용 DB -----------------
    def init_database(self):
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
            print("✅ 로컬 분석 DB 초기화 완료")
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"❌ 로컬 DB 초기화 오류: {e}")
            messagebox.showerror("데이터베이스 오류", f"데이터베이스 초기화 실패:\n{e}")
        finally:
            if conn:
                conn.close()

    # ----------------- 폰트/레이아웃/UI -----------------
    def setup_korean_font(self):
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
        self.setup_header()
        main_container = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_container.pack(fill='both', expand=True, padx=10, pady=5)
        left_panel = tk.Frame(main_container, bg=self.colors['bg_medium'], width=380, relief='solid', bd=1)
        left_panel.pack(side='left', fill='y', padx=(0, 5))
        left_panel.pack_propagate(False)
        self.setup_status_cards(left_panel)
        right_panel = tk.Frame(main_container, bg=self.colors['bg_dark'])
        right_panel.pack(side='right', fill='both', expand=True)
        self.setup_plots(right_panel)
        self.setup_statusbar()

    def setup_header(self):
        header = tk.Frame(self.root, bg=self.colors['bg_medium'], relief='solid', bd=1)
        header.pack(fill='x', padx=0, pady=0)
        title_frame = tk.Frame(header, bg=self.colors['bg_medium'])
        title_frame.pack(side='left', padx=20, pady=10)
        tk.Label(title_frame, text="🏭 SMART FACTORY",
                 font=(self.font_family, 24, 'bold'),
                 bg=self.colors['bg_medium'], fg=self.colors['text_primary']).pack(anchor='w')
        tk.Label(title_frame, text="IMU Real-Time Monitoring & Analysis System",
                 font=(self.font_family, 12),
                 bg=self.colors['bg_medium'], fg=self.colors['text_secondary']).pack(anchor='w')

        button_frame = tk.Frame(header, bg=self.colors['bg_medium'])
        button_frame.pack(side='left', fill='both', expand=True, padx=20)
        btn_cfg = {'font': (self.font_family, 11, 'bold'), 'relief': 'raised',
                   'cursor': 'hand2', 'bd': 2, 'height': 2, 'padx': 20, 'pady': 5}
        btn_container = tk.Frame(button_frame, bg=self.colors['bg_medium'])
        btn_container.pack(expand=True)

        self.auto_btn = tk.Button(btn_container, text="🚀 AUTO MEASUREMENT",
                                  command=self.start_auto_collection,
                                  bg=self.colors['success'], fg='white',
                                  activebackground='#00a876', **btn_cfg)
        self.auto_btn.pack(side='left', padx=3, pady=10)

        # ✅ 로그인 버튼 (커스텀 다이얼로그 사용)
        tk.Button(btn_container, text="🔐 LOGIN", command=self.login_via_api,
                  bg='#343a40', fg='white',
                  activebackground='#23272b', **btn_cfg).pack(side='left', padx=3, pady=10)

        tk.Button(btn_container, text="🤖 LOAD MODEL", command=self.load_model,
                  bg=self.colors['info'], fg='white',
                  activebackground='#138496', **btn_cfg).pack(side='left', padx=3, pady=10)

        tk.Button(btn_container, text="💾 SAVE EXCEL", command=self.save_data,
                  bg=self.colors['warning'], fg='white',
                  activebackground='#e68900', **btn_cfg).pack(side='left', padx=3, pady=10)

        tk.Button(btn_container, text="🗄️ SAVE DB", command=self.save_to_database,
                  bg='#6f42c1', fg='white',
                  activebackground='#5a32a3', **btn_cfg).pack(side='left', padx=3, pady=10)

        tk.Button(btn_container, text="🔄 RESET", command=self.clear_data,
                  bg=self.colors['text_secondary'], fg='white',
                  activebackground='#5a6268', **btn_cfg).pack(side='left', padx=3, pady=10)

        self.time_label = tk.Label(header, text="", font=(self.font_family, 15, 'bold'),
                                   bg=self.colors['bg_medium'], fg=self.colors['info'])
        self.time_label.pack(side='right', padx=20)
        self.update_time()

    def update_time(self):
        now = datetime.now()
        self.time_label.config(text=f"🕒 {now.strftime('%Y-%m-%d %H:%M:%S')}")
        self.root.after(1000, self.update_time)

    def setup_status_cards(self, parent):
        tk.Frame(parent, bg=self.colors['bg_medium'], height=10).pack()

        conn_card = self.create_card(parent, "CONNECTION STATUS")
        self.conn_indicator = self.create_status_indicator(conn_card, "WebSocket", "OFFLINE")

        model_card = self.create_card(parent, "AI MODEL STATUS")
        self.model_indicator = self.create_status_indicator(model_card, "ML Pipeline", "NOT LOADED")

        data_card = self.create_card(parent, "DATA COLLECTION")
        self.data_count_label = tk.Label(data_card, text="Records: 0",
                                         font=(self.font_family, 15, 'bold'),
                                         bg=self.colors['bg_light'], fg=self.colors['text_primary'])
        self.data_count_label.pack(pady=5)
        self.session_label = tk.Label(data_card, text="Session: N/A",
                                      font=(self.font_family, 13, 'bold'),
                                      bg=self.colors['bg_light'], fg=self.colors['text_secondary'])
        self.session_label.pack()

        measure_card = self.create_card(parent, "MEASUREMENT")
        self.measure_status = tk.Label(measure_card, text="⏸ STANDBY",
                                       font=(self.font_family, 18, 'bold'),
                                       bg=self.colors['bg_light'], fg=self.colors['text_secondary'])
        self.measure_status.pack(pady=10)
        self.countdown_label = tk.Label(measure_card, text="",
                                        font=(self.font_family, 28, 'bold'),
                                        bg=self.colors['bg_light'], fg=self.colors['warning'])
        self.countdown_label.pack()

        threshold_card = self.create_card(parent, "FAULT THRESHOLD")
        tfrm = tk.Frame(threshold_card, bg=self.colors['bg_light']); tfrm.pack(pady=10)
        tk.Label(tfrm, text="Drift Limit:", font=(self.font_family, 14, 'bold'),
                 bg=self.colors['bg_light'], fg=self.colors['text_secondary']).pack(side='left', padx=5)
        self.threshold_display = tk.Label(tfrm, text=f"{self.threshold}°",
                                          font=(self.font_family, 20, 'bold'),
                                          bg=self.colors['bg_light'], fg=self.colors['accent'])
        self.threshold_display.pack(side='left')

        # === 업로드 설정 카드 ===
        upload_card = self.create_card(parent, "UPLOAD SETTINGS")
        frm = tk.Frame(upload_card, bg=self.colors['bg_light'])
        frm.pack(fill='x', padx=8, pady=8)

        # ✅ API Base
        tk.Label(frm, text="API Base", font=(self.font_family, 11, 'bold'),
                 bg=self.colors['bg_light'], fg=self.colors['text_secondary']).grid(row=0, column=0, sticky='w')
        tk.Entry(frm, textvariable=self.api_base_var, width=28,
                 font=(self.font_family, 11)).grid(row=0, column=1, sticky='we', pady=2)

        # DB URL (SQLite)
        tk.Label(frm, text="DB URL (SQLite)", font=(self.font_family, 11, 'bold'),
                 bg=self.colors['bg_light'], fg=self.colors['text_secondary']).grid(row=1, column=0, sticky='w')
        tk.Entry(frm, textvariable=self.db_url_var, width=28,
                 font=(self.font_family, 11)).grid(row=1, column=1, sticky='we', pady=2)

        # Operator (로그인 시 갱신)
        tk.Label(frm, text="Operator", font=(self.font_family, 11, 'bold'),
                 bg=self.colors['bg_light'], fg=self.colors['text_secondary']).grid(row=2, column=0, sticky='w')
        tk.Label(frm, textvariable=self.operator_name_var, font=(self.font_family, 11, 'bold'),
                 bg=self.colors['bg_light'], fg=self.colors['text_primary']).grid(row=2, column=1, sticky='w', pady=2)

        # Destination (고정 표기: Room3)
        tk.Label(frm, text="Destination", font=(self.font_family, 11, 'bold'),
                 bg=self.colors['bg_light'], fg=self.colors['text_secondary']).grid(row=3, column=0, sticky='w')
        tk.Label(frm, text="Room3", font=(self.font_family, 11, 'bold'),
                 bg=self.colors['bg_light'], fg=self.colors['text_primary']).grid(row=3, column=1, sticky='w', pady=2)

        # Box No (숫자)
        tk.Label(frm, text="Box No", font=(self.font_family, 11, 'bold'),
                 bg=self.colors['bg_light'], fg=self.colors['text_secondary']).grid(row=4, column=0, sticky='w')
        tk.Entry(frm, textvariable=self.upload_box_no_var, width=28,
                 font=(self.font_family, 11)).grid(row=4, column=1, sticky='we', pady=2)

        for i in range(2):
            frm.grid_columnconfigure(i, weight=1)

    def create_card(self, parent, title):
        card = tk.Frame(parent, bg=self.colors['bg_light'], relief='solid', bd=1)
        card.pack(fill='x', padx=10, pady=5)
        tk.Frame(card, bg=self.colors['accent'], height=3).pack(fill='x')
        tk.Label(card, text=title, font=(self.font_family, 14, 'bold'),
                 bg=self.colors['bg_light'], fg=self.colors['text_secondary']).pack(pady=(10, 5))
        return card

    def create_status_indicator(self, parent, label, status):
        frame = tk.Frame(parent, bg=self.colors['bg_light']); frame.pack(pady=10)
        self.led_canvas = tk.Canvas(frame, width=24, height=24, bg=self.colors['bg_light'], highlightthickness=0)
        self.led_canvas.pack(side='left', padx=5)
        led = self.led_canvas.create_oval(2, 2, 22, 22, fill=self.colors['danger'], outline=self.colors['danger'])
        status_label = tk.Label(frame, text=f"{label}: {status}",
                                font=(self.font_family, 14, 'bold'),
                                bg=self.colors['bg_light'], fg=self.colors['text_primary'])
        status_label.pack(side='left')
        return {'canvas': self.led_canvas, 'led': led, 'label': status_label}

    def setup_plots(self, parent):
        plot_frame = tk.Frame(parent, bg=self.colors['bg_medium'], relief='solid', bd=1)
        plot_frame.pack(fill='both', expand=True, padx=5, pady=5)
        self.fig, axs = plt.subplots(4, 2, figsize=(14, 10), facecolor='white')
        self.axes = axs.flatten()
        for i, ax in enumerate(self.axes):
            ax.set_facecolor('#fafafa')
            ax.set_title(f"[SENSOR {i}]", fontsize=12, fontweight='bold',
                         color=self.colors['text_primary'], fontfamily=self.font_family, pad=10)
            ax.set_ylabel("Angle (°)", fontsize=10, color=self.colors['text_secondary'], fontfamily=self.font_family)
            ax.grid(True, alpha=0.3, color=self.colors['grid'], linestyle='-', linewidth=0.5)
            ax.tick_params(colors=self.colors['text_secondary'])
            for spine in ax.spines.values():
                spine.set_edgecolor('#495057'); spine.set_linewidth(1.5); spine.set_capstyle('round')
            ax.patch.set_edgecolor('#343a40'); ax.patch.set_linewidth(2)
        self.axes[6].set_xlabel("Time", fontsize=10, color=self.colors['text_secondary'], fontfamily=self.font_family)
        self.axes[7].set_xlabel("Time", fontsize=10, color=self.colors['text_secondary'], fontfamily=self.font_family)
        plt.tight_layout(pad=3.0, h_pad=2.5, w_pad=2.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def setup_statusbar(self):
        statusbar = tk.Frame(self.root, bg=self.colors['bg_medium'], height=35, relief='solid', bd=1)
        statusbar.pack(side='bottom', fill='x'); statusbar.pack_propagate(False)
        self.status_message = tk.Label(statusbar, text="✅ System Ready",
                                       font=(self.font_family, 11, 'bold'),
                                       bg=self.colors['bg_medium'], fg=self.colors['text_secondary'])
        self.status_message.pack(side='left', padx=20)
        info_frame = tk.Frame(statusbar, bg=self.colors['bg_medium']); info_frame.pack(side='right', padx=20)
        tk.Label(info_frame, text="v2.2 | Local SQLite Upload",
                 font=(self.font_family, 10, 'bold'),
                 bg=self.colors['bg_medium'], fg=self.colors['text_secondary']).pack(side='right')

    # ----------------- 공용 UI 유틸 -----------------
    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def update_status(self, message, status_type='info'):
        colors = {'info': self.colors['info'], 'success': self.colors['success'],
                  'warning': self.colors['warning'], 'danger': self.colors['danger']}
        icons = {'info': 'ℹ️', 'success': '✅', 'warning': '⚠️', 'danger': '❌'}
        self.status_message.config(text=f"{icons.get(status_type,'')} {message}",
                                   fg=colors.get(status_type, self.colors['text_secondary']))
        self.root.update()

    def update_connection_status(self, connected):
        if connected:
            self.conn_indicator['canvas'].itemconfig(self.conn_indicator['led'],
                                                     fill=self.colors['success'], outline=self.colors['success'])
            self.conn_indicator['label'].config(text="WebSocket: ONLINE", fg=self.colors['success'])
        else:
            self.conn_indicator['canvas'].itemconfig(self.conn_indicator['led'],
                                                     fill=self.colors['danger'], outline=self.colors['danger'])
            self.conn_indicator['label'].config(text="WebSocket: OFFLINE", fg=self.colors['danger'])

    def update_model_status(self, loaded):
        if loaded:
            self.model_indicator['canvas'].itemconfig(self.model_indicator['led'],
                                                      fill=self.colors['success'], outline=self.colors['success'])
            self.model_indicator['label'].config(text="ML Pipeline: READY", fg=self.colors['success'])
        else:
            self.model_indicator['canvas'].itemconfig(self.model_indicator['led'],
                                                      fill=self.colors['danger'], outline=self.colors['danger'])
            self.model_indicator['label'].config(text="ML Pipeline: NOT LOADED", fg=self.colors['danger'])

    def update_data_count(self):
        with self.data_lock:
            count = len(self.data_records)
        self.data_count_label.config(text=f"Records: {count:,}")
        if self.session_id:
            self.session_label.config(text=f"Session: {self.session_id[:8]}...")

    # ----------------- 측정/스트리밍 -----------------
    def wait_for_connection(self, callback):
        start_time = datetime.now()
        def check():
            elapsed = (datetime.now() - start_time).total_seconds()
            if self.ws_connected:
                callback(True)
            elif elapsed >= self.connection_timeout:
                callback(False)
            else:
                self.root.after(100, check)
        # 첫 호출도 after로 UI 루프와 동기화
        self.root.after(100, check)

    def start_auto_collection(self):
        if self.pipeline is None:
            messagebox.showerror("모델 오류", "먼저 AI 모델을 로드해주세요!")
            return
        self.auto_mode = True
        self._countdown_started = False
        with self.data_lock:
            self.data_records = []
        self.predictions_data = {}
        self.collection_start_time = datetime.now()
        self.session_id = str(uuid.uuid4())
        self.save_session_info()
        self.measure_status.config(text="CONNECTING...", fg=self.colors['warning'])
        self.update_status("WebSocket 연결 중...", 'warning')
        self.start_stream()

        # 백업: 일정시간 내 연결 확인
        def on_conn(ok):
            def _on_main():
                if ok:
                    self.measure_status.config(text="COLLECTING DATA", fg=self.colors['success'])
                else:
                    self.auto_mode = False
                    self.stop_stream()
                    self.measure_status.config(text="CONNECTION FAILED", fg=self.colors['danger'])
                    self.update_status("WebSocket 서버 연결 실패", 'danger')
                    messagebox.showerror("연결 오류", "WebSocket 서버에 연결할 수 없습니다.\n서버 상태를 확인해주세요.")
            self.root.after(0, _on_main)
        self.wait_for_connection(on_conn)

    def save_session_info(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("BEGIN")
            conn.execute('''
                INSERT INTO measurement_sessions 
                (session_id, start_time, session_type, operator_name, facility_location, equipment_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (self.session_id, self.collection_start_time.isoformat(),
                  "자동", "운영자", "시설위치", "IMU-001"))
            conn.commit()
            self.update_data_count()
        except Exception as e:
            if conn: conn.rollback()
            print(f"세션 정보 저장 오류: {e}")
        finally:
            if conn: conn.close()

    def start_countdown(self, seconds_left):
        if seconds_left > 0 and self.auto_mode:
            self.countdown_label.config(text=f"{seconds_left}")
            self.root.after(1000, lambda: self.start_countdown(seconds_left-1))
        elif self.auto_mode:
            self.countdown_label.config(text="")
            self.measure_status.config(text="🔍 ANALYZING...", fg=self.colors['info'])
            self.stop_stream()
            # 다음 사이클 대비 초기화는 predict() 시작부에서 처리
            self.root.after(500, self.predict)

    def clear_data(self):
        with self.data_lock:
            self.data_records = []
        self.predictions_data = {}
        self.update_data_count()
        for ax in self.axes:
            ax.cla(); ax.set_facecolor('#fafafa')
            ax.set_title(f"[SENSOR {self.axes.tolist().index(ax)}]", fontsize=12, fontweight='bold',
                         color=self.colors['text_primary'], fontfamily=self.font_family, pad=10)
            ax.set_ylabel("Angle (°)", fontsize=10, color=self.colors['text_secondary'], fontfamily=self.font_family)
            ax.grid(True, alpha=0.3, color=self.colors['grid'], linestyle='-', linewidth=0.5)
            ax.tick_params(colors=self.colors['text_secondary'])
            for s in ax.spines.values():
                s.set_edgecolor('#495057'); s.set_linewidth(1.5); s.set_capstyle('round')
            ax.patch.set_edgecolor('#343a40'); ax.patch.set_linewidth(2)
        self.canvas.draw()
        self.measure_status.config(text="⏸ STANDBY", fg=self.colors['text_secondary'])
        self.update_status("데이터 초기화 완료", 'success')

    # --- ✅ 모든 웹소켓 콜백에서 UI 접근은 메인 스레드로 던지기 ---
    def on_message(self, ws, message):
        try:
            msg = json.loads(message); ts = datetime.now()
            with self.data_lock:
                if 'sensors' in msg:
                    for sensor in msg['sensors']:
                        rec = sensor.copy(); rec['SN'] = rec.get('id'); rec['timestamp'] = ts
                        self.data_records.append(rec)
                else:
                    msg['timestamp'] = ts; self.data_records.append(msg)
                if len(self.data_records) > self.MAX_RECORDS:
                    self.data_records = self.data_records[-self.MAX_RECORDS:]
        except Exception as e:
            print("메시지 파싱 오류:", e)
        finally:
            # UI 라벨 갱신은 메인 스레드
            self.root.after(0, self.update_data_count)
            # 백업 트리거: 최초 데이터 수신 시 카운트다운 시작
            def _kickoff():
                if self.auto_mode and not self._countdown_started:
                    self._countdown_started = True
                    self.measure_status.config(text="COLLECTING DATA", fg=self.colors['success'])
                    self.update_status("자동 측정 진행 중 (5초)", 'success')
                    self.start_countdown(5)
            self.root.after(0, _kickoff)

    def on_error(self, ws, error):
        def _on_main():
            print("WebSocket 오류:", error)
            self.ws_connected = False
            self.update_connection_status(False)
            self.update_status("연결 오류 발생", 'danger')
        self.root.after(0, _on_main)

    def on_close(self, ws, close_status, close_msg):
        def _on_main():
            print("WebSocket 연결 종료")
            self.ws_connected = False
            self.update_connection_status(False)
            if self.streaming:
                self.update_status("연결이 끊어졌습니다", 'warning')
        self.root.after(0, _on_main)

    def on_open(self, ws):
        def _on_main():
            print("WebSocket 연결 성공")
            self.ws_connected = True
            self.update_connection_status(True)
            self.update_status("데이터 수집 중", 'success')
            if self.auto_mode and not self._countdown_started:
                self._countdown_started = True
                self.measure_status.config(text="COLLECTING DATA", fg=self.colors['success'])
                self.update_status("자동 측정 진행 중 (5초)", 'success')
                self.start_countdown(5)
        self.root.after(0, _on_main)

    def start_stream(self):
        if self.streaming: return
        self.ws_connected = False
        ws_url = "ws://10.200.246.81:81"
        try:
            self.ws = websocket.WebSocketApp(ws_url,
                                             on_open=self.on_open,
                                             on_message=self.on_message,
                                             on_error=self.on_error,
                                             on_close=self.on_close)
            self.wst = threading.Thread(target=self.ws.run_forever); self.wst.daemon = True
            self.streaming = True; self.wst.start()
            self.root.after(100, self.update_plot)
        except Exception as e:
            print(f"WebSocket 시작 오류: {e}"); self.streaming = False; self.ws_connected = False

    def stop_stream(self):
        if not self.streaming: return
        self.streaming = False; self.ws_connected = False
        self.update_connection_status(False)
        try: self.ws.close()
        except: pass
        if self.auto_mode:
            self.countdown_label.config(text="")
        else:
            self.update_status("데이터 수집 중지", 'info')

    def update_plot(self):
        if not self.streaming: return
        try:
            with self.data_lock:
                if self.data_records:
                    df = pd.DataFrame(self.data_records.copy())
            if 'df' in locals() and not df.empty:
                if 'timestamp' not in df.columns:
                    self.root.after(100, self.update_plot); return
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                for ax in self.axes:
                    ax.cla(); ax.set_facecolor('#fafafa')
                    ax.grid(True, alpha=0.3, color=self.colors['grid'], linestyle='-', linewidth=0.5)
                    ax.tick_params(colors=self.colors['text_secondary'])
                    for s in ax.spines.values():
                        s.set_edgecolor('#495057'); s.set_linewidth(1.5); s.set_capstyle('round')
                    ax.patch.set_edgecolor('#343a40'); ax.patch.set_linewidth(2)
                for sn in range(8):
                    ax = self.axes[sn]
                    ax.set_title(f"[SENSOR {sn}]", fontsize=12, fontweight='bold',
                                 color=self.colors['text_primary'], fontfamily=self.font_family, pad=10)
                    if 'SN' in df.columns:
                        sub = df[df['SN'] == sn]
                        if not sub.empty and all(c in sub.columns for c in ['ROLL','PITCH','YAW']):
                            ax.plot(sub['timestamp'], sub['ROLL'], '#dc3545', linewidth=2, label='Roll', alpha=0.8)
                            ax.plot(sub['timestamp'], sub['PITCH'], '#28a745', linewidth=2, label='Pitch', alpha=0.8)
                            ax.plot(sub['timestamp'], sub['YAW'], '#007bff', linewidth=2, label='Yaw', alpha=0.8)
                            ax.legend(loc='upper right', fontsize=9, framealpha=0.95,
                                      facecolor='white', edgecolor=self.colors['grid'],
                                      prop={'family': self.font_family})
                    ax.set_ylabel("Angle (°)", fontsize=10, color=self.colors['text_secondary'], fontfamily=self.font_family)
                    if sn >= 6:
                        ax.set_xlabel("Time", fontsize=10, color=self.colors['text_secondary'], fontfamily=self.font_family)
                plt.tight_layout(pad=3.0, h_pad=2.5, w_pad=2.5)
                self.canvas.draw()
        except Exception as e:
            print(f"플롯 업데이트 오류: {e}")
        self.root.after(100, self.update_plot)

    def save_data(self):
        with self.data_lock:
            if not self.data_records:
                messagebox.showwarning("경고", "저장할 데이터가 없습니다"); return
            df = pd.DataFrame(self.data_records.copy())
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel 파일","*.xlsx")])
        if file_path:
            try:
                df.to_excel(file_path, index=False)
                self.update_status(f"데이터 저장 완료: {os.path.basename(file_path)}", 'success')
                messagebox.showinfo("성공", f"데이터가 저장되었습니다:\n{file_path}")
            except Exception as e:
                self.update_status("파일 저장 실패", 'danger')
                messagebox.showerror("오류", f"파일 저장 실패:\n{e}")

    # ----------------- API 로그인 -----------------
    def _get_api_base(self):
        url = (self.api_base_var.get() or "").strip().rstrip("/")
        return url or "http://127.0.0.1:8000"

    def login_via_api(self):
        # 커스텀 다이얼로그로 이메일/비밀번호를 한 번에 입력
        dlg = LoginDialog(self.root, "로그인")
        email = dlg.email
        password = dlg.password
        if not email or password is None:
            # 사용자가 취소를 눌렀거나 입력이 없을 때
            return

        base = self._get_api_base()
        try:
            self.update_status("로그인 중...", "info")
            resp = requests.post(f"{base}/auth/login",
                                 json={"email": email, "password": password},
                                 timeout=10)
            if resp.status_code != 200:
                try:
                    msg = resp.json().get("detail", resp.text)
                except Exception:
                    msg = resp.text
                self.update_status("로그인 실패", "danger")
                messagebox.showerror("로그인 실패", f"{resp.status_code}: {msg}")
                return

            data = resp.json()
            self.auth_token = data.get("token")
            self.user_summary = data.get("summary", {})
            display_name = self.user_summary.get("name") or self.user_summary.get("email") or "user"
            self.operator_name_var.set(display_name)

            self.update_status(f"로그인 성공: {display_name}", "success")
            messagebox.showinfo("성공", f"{display_name}님, 환영합니다!\n로그인 토큰이 설정되었습니다.")
        except Exception as e:
            self.update_status("로그인 중 오류", "danger")
            messagebox.showerror("오류", f"로그인 실패:\n{e}")

    # ----------------- 숫자 정규화 유틸 -----------------
    @staticmethod
    def _finite_or_none(x):
        try:
            v = float(x)
            if np.isfinite(v):
                return v
            return None
        except Exception:
            return None

    # ----------------- 로컬 SQLite 업로드 + API 업로드 -----------------
    @staticmethod
    def _sqlite_path_from_url(url_text: str) -> str:
        """
        'sqlite:///./smartfactory.db' 형태 또는 파일 경로를 받아 실제 파일 경로로 변환
        """
        url_text = (url_text or "").strip()
        if not url_text:
            return "smartfactory.db"
        lower = url_text.lower()
        if lower.startswith("sqlite:///"):
            path = url_text[10:]  # after sqlite:///
        elif lower.startswith("sqlite://"):
            path = url_text[9:]
        else:
            path = url_text
        path = os.path.expanduser(path)
        return path

    @staticmethod
    def _table_exists(conn, table_name: str) -> bool:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND lower(name)=lower(?)", (table_name,))
        return cur.fetchone() is not None

    def _ensure_min_schema(self, conn):
        """
        최소 스키마 보장:
        - user(id, email, password_hash, name, role, created_at, updated_at)
        - imurecord(id, code, serial, inspected_at, passed, inspector_id, box_no, destination, arrived, roll, pitch, yaw, created_at, updated_at)
        이미 존재하면 건너뜀(기존 스키마와 충돌하지 않도록 NOT EXISTS 사용)
        """
        conn.execute("PRAGMA foreign_keys=ON")
        if not self._table_exists(conn, "user"):
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE,
                    password_hash TEXT,
                    name TEXT,
                    role TEXT DEFAULT 'USER',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        if not self._table_exists(conn, "imurecord"):
            conn.execute("""
                CREATE TABLE IF NOT EXISTS imurecord (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT UNIQUE,
                    serial TEXT,
                    inspected_at DATETIME,
                    passed INTEGER,
                    inspector_id INTEGER,
                    box_no INTEGER,
                    destination TEXT,
                    arrived INTEGER,
                    roll REAL,
                    pitch REAL,
                    yaw REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(inspector_id) REFERENCES user(id)
                )
            """)

    def _get_or_create_admin_id(self, conn) -> int:
        cur = conn.execute("SELECT id FROM user WHERE lower(name)=lower(?) OR lower(email)=lower(?)",
                           ("admin", "admin@local"))
        row = cur.fetchone()
        if row:
            return row[0]
        conn.execute("INSERT INTO user (email, password_hash, name, role) VALUES (?,?,?,?)",
                     ("admin@local", "local", "admin", "ADMIN"))
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def _next_imu_code(self, conn) -> str:
        cur = conn.execute("SELECT COALESCE(MAX(CAST(code AS INTEGER)), 0) FROM imurecord WHERE code GLOB '[0-9]*'")
        max_n = cur.fetchone()[0] or 0
        n = int(max_n) + 1
        width = 3 if n < 1000 else len(str(n))
        return str(n).zfill(width)

    def save_to_database(self):
        """
        저장 동작:
        - ✅ 로그인(토큰 보유) 상태면: API POST /imu 로 업로드 (inspector_id는 로그인 사용자로 자동 반영)
        - 비로그인 상태면: 기존 로컬 SQLite에 직접 insert (레거시 호환)
        """
        with self.data_lock:
            has_data = bool(self.data_records)
        if not has_data:
            messagebox.showwarning("경고", "업로드할 데이터가 없습니다"); return
        if not self.predictions_data:
            messagebox.showwarning("경고", "예측 결과가 없습니다. 먼저 자동 측정을 실행하거나 예측을 완료하세요.")
            return

        # 공통 입력값
        inspected_at = (self.collection_start_time.isoformat()
                        if self.collection_start_time else datetime.utcnow().isoformat())
        destination = "Room3"
        arrived = True
        box_no = None
        box_no_text = (self.upload_box_no_var.get() or "").strip()
        if box_no_text:
            try:
                box_no = int(box_no_text)
            except ValueError:
                messagebox.showwarning("입력 오류", "Box No는 숫자여야 합니다.")
                return

        # ✅ 1) 로그인 상태면 API 업로드
        if self.auth_token:
            base = self._get_api_base()
            headers = {"X-Auth-Token": self.auth_token}
            success, failed = 0, 0
            failures = []

            self.update_status("API로 업로드 중...", "info")
            for sensor_id, pred in sorted(self.predictions_data.items()):
                try:
                    payload = {
                        "serial": f"SENSOR-{sensor_id:02d}",
                        "inspected_at": inspected_at,
                        "passed": (not bool(pred.get("is_faulty", False))),
                        "box_no": box_no,
                        "destination": destination,
                        "arrived": arrived,
                        "roll":  self._finite_or_none(pred.get("roll_drift")),
                        "pitch": self._finite_or_none(pred.get("pitch_drift")),
                        "yaw":   self._finite_or_none(pred.get("yaw_drift")),
                    }
                    print("UPLOADING:", sensor_id, payload)  # 디버그 로그

                    resp = requests.post(f"{base}/imu", json=payload, headers=headers, timeout=10)
                    if resp.status_code in (200, 201):
                        success += 1
                    else:
                        try:
                            msg = resp.json().get("detail", resp.text)
                        except Exception:
                            msg = resp.text
                        failed += 1
                        failures.append(f"센서 {sensor_id}: {resp.status_code} {msg}")
                except Exception as e:
                    failed += 1
                    failures.append(f"센서 {sensor_id}: {e}")

            if failed == 0:
                self.update_status(f"API 업로드 완료({success}건)", "success")
                messagebox.showinfo("성공", f"API 업로드 완료!\n- 업로드 성공: {success}건")
            else:
                self.update_status(f"일부 업로드 실패: 성공 {success} / 실패 {failed}", "warning")
                detail = "\n".join(failures[:5]) + ("\n..." if len(failures) > 5 else "")
                messagebox.showwarning("부분 실패",
                                       f"일부 업로드에 실패했습니다.\n- 성공: {success}\n- 실패: {failed}\n\n상세:\n{detail}")
            return

        # 🔁 2) 비로그인 상태: 로컬 SQLite (레거시)
        db_path = self._sqlite_path_from_url(self.db_url_var.get())
        try:
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        except Exception:
            pass

        try:
            conn = sqlite3.connect(db_path, timeout=30)
            conn.execute("BEGIN")
            self._ensure_min_schema(conn)
            admin_id = self._get_or_create_admin_id(conn)   # 레거시: admin 계정
            conn.commit()
        except Exception as e:
            try: conn.rollback()
            except: pass
            messagebox.showerror("DB 오류", f"DB 초기화 실패:\n{e}")
            return

        success, failed = 0, 0
        failures = []
        try:
            conn.execute("BEGIN")
            for sensor_id, pred in sorted(self.predictions_data.items()):
                try:
                    passed = 0 if bool(pred.get("is_faulty", False)) else 1
                    roll_val = self._finite_or_none(pred.get("roll_drift"))
                    pitch_val = self._finite_or_none(pred.get("pitch_drift"))
                    yaw_val = self._finite_or_none(pred.get("yaw_drift"))
                    serial = f"SENSOR-{sensor_id:02d}"
                    code = self._next_imu_code(conn)
                    conn.execute("""
                        INSERT INTO imurecord
                        (code, serial, inspected_at, passed, inspector_id, box_no, destination, arrived, roll, pitch, yaw, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """, (code, serial, inspected_at, passed, admin_id, box_no, destination, 1 if True else 0,
                          roll_val, pitch_val, yaw_val))
                    success += 1
                except Exception as ie:
                    failed += 1
                    failures.append(f"센서 {sensor_id}: {ie}")
            conn.commit()
        except Exception as e:
            try: conn.rollback()
            except: pass
            messagebox.showerror("DB 오류", f"업로드 중 오류:\n{e}")
            conn.close()
            return
        finally:
            conn.close()

        if failed == 0:
            self.update_status(f"로컬 DB 업로드 완료({success}건) → {db_path}", "success")
            messagebox.showinfo("성공", f"로컬 DB 업로드 완료!\n- 업로드 성공: {success}건\n- DB: {db_path}")
        else:
            self.update_status(f"일부 업로드 실패: 성공 {success} / 실패 {failed}", "warning")
            detail = "\n".join(failures[:5]) + ("\n..." if len(failures) > 5 else "")
            messagebox.showwarning("부분 실패",
                                   f"일부 업로드에 실패했습니다.\n- 성공: {success}\n- 실패: {failed}\n\n상세:\n{detail}")

    # ----------------- 모델/예측 -----------------
    def load_model(self):
        file_path = filedialog.askopenfilename(title="AI 모델 파일 선택", filetypes=[("Pickle 파일","*.pkl")])
        if not file_path: return
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
        # 다음 자동 사이클을 위해 플래그 리셋
        self._countdown_started = False

        if self.pipeline is None:
            messagebox.showerror("오류", "AI 모델이 로드되지 않았습니다"); return
        with self.data_lock:
            if not self.data_records:
                messagebox.showwarning("경고", "예측할 데이터가 없습니다"); return
            data_for_prediction = self.data_records.copy()
        try:
            df = pd.DataFrame(data_for_prediction)
            required = ['timestamp','SN','ROLL','PITCH','YAW','X_DEL_ANG','Y_DEL_ANG','Z_DEL_ANG']
            missing = [c for c in required if c not in df.columns]
            if missing:
                messagebox.showerror("오류", f"필수 데이터 컬럼이 없습니다: {missing}"); return
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            predictions = {}; self.predictions_data = {} 

            for sn in range(8):
                sub = df[df['SN'] == sn].sort_values('timestamp')
                if len(sub) < 2: continue
                t0 = sub['timestamp'].iloc[0]                             
                start_time = t0 + timedelta(seconds=1); end_time = start_time + timedelta(seconds=4)
                window = sub[(sub['timestamp'] >= start_time) & (sub['timestamp'] <= end_time)]
                if len(window) < 2: continue

                p5 = (-window['X_DEL_ANG']).mean()
                q5 = (-window['Z_DEL_ANG']).mean()
                r5 = ( window['Y_DEL_ANG']).mean()

                dt = (window['timestamp'].iloc[-1] - window['timestamp'].iloc[0]).total_seconds()
                if dt == 0: continue
                Rd5 = (window['ROLL'].iloc[-1]-window['ROLL'].iloc[0])/dt
                Pd5 = (window['PITCH'].iloc[-1]-window['PITCH'].iloc[0])/dt
                Yd5 = (window['YAW'].iloc[-1]-window['YAW'].iloc[0])/dt

                R = window['ROLL'].values; P = window['PITCH'].values
                p = -window['X_DEL_ANG'].values; q = -window['Z_DEL_ANG'].values; r = window['Y_DEL_ANG'].values
                
                R_rad = np.deg2rad(R); P_rad = np.deg2rad(P)
                cos_P = np.cos(P_rad); cos_P = np.where(np.abs(cos_P) < 1e-10, 1e-10*np.sign(cos_P), cos_P)
                sin_R = np.sin(R_rad); cos_R = np.cos(R_rad)
                tan_P = np.clip(np.tan(P_rad), -100, 100)
                Rdot = p + (q*sin_R + r*cos_R) * tan_P
                Pdot = q * cos_R - r * sin_R
                Ydot = (q * sin_R + r * cos_R) / cos_P
                Rdot = np.nan_to_num(Rdot); Pdot = np.nan_to_num(Pdot); Ydot = np.nan_to_num(Ydot)
                Rdot5 = Rdot.mean(); Pdot5 = Pdot.mean(); Ydot5 = Ydot.mean()

                X_feat = pd.DataFrame([[p5,q5,r5,Rd5,Pd5,Yd5,Rdot5,Pdot5,Ydot5]],
                                      columns=['p5','q5','r5','Rd5','Pd5','Yd5','Rdot5','Pdot5','Ydot5'])
                if X_feat.isnull().any().any(): continue

                try:
                    raw_pred = self.pipeline.predict(X_feat)
                except Exception as e:
                    print(f"센서 {sn} 예측 호출 오류: {e}"); continue

                # --- ✅ 출력 평탄화/검증: [[Rdel,Pdel,Ydel]] 등 어떤 형태든 안전하게 3개 추출
                pred_flat = np.array(raw_pred).reshape(-1)

                if pred_flat.size >= 3 and np.all(np.isfinite(pred_flat[:3])):
                    r_pred, p_pred, y_pred = map(float, pred_flat[:3])
                else:
                    # 3개 미만 또는 비유한수 → 이 센서는 스킵
                    continue

                # 결과 저장 (3축 기준)
                predictions[sn] = [r_pred, p_pred, y_pred]

                drift_vals = {'Roll': abs(r_pred), 'Pitch': abs(p_pred), 'Yaw': abs(y_pred)}
                max_axis = max(drift_vals, key=drift_vals.get); max_val = drift_vals[max_axis]
                max_signed = {'Roll': r_pred, 'Pitch': p_pred, 'Yaw': y_pred}[max_axis]
                is_faulty = max_val > self.threshold; status = "고장" if is_faulty else "정상"

                self.predictions_data[sn] = {
                    'roll_drift': r_pred,
                    'pitch_drift': p_pred,
                    'yaw_drift': y_pred,
                    'max_drift_axis': max_axis,
                    'max_drift_value': float(max_val),
                    'max_drift_signed': float(max_signed),
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
        for sn, ax in enumerate(self.axes):
            pred = predictions.get(sn)
            for txt in list(ax.texts): txt.remove()
            if not pred:
                label = "DATA\nINSUFFICIENT"; color = self.colors['text_secondary']
                bgcolor = '#f8f9fa'; border_color = self.colors['grid']
            else:
                if len(pred) == 3:
                    r_pred, p_pred, y_pred = pred
                    drift = {'Roll': abs(r_pred), 'Pitch': abs(p_pred), 'Yaw': abs(y_pred)}
                    max_axis = max(drift, key=drift.get); max_val = max(drift.values())
                    max_signed = r_pred if max_axis=='Roll' else p_pred if max_axis=='Pitch' else y_pred
                    fail = max_val > self.threshold
                    status = "[FAULT]" if fail else "[NORMAL]"
                    color = 'white'
                    bgcolor = self.colors['danger'] if fail else self.colors['success']
                    border_color = '#dc3545' if fail else '#28a745'
                    label = f"100s DRIFT PREDICTION\n{max_axis}: {max_signed:.2f}°\n{status}"
                else:
                    # 호환 유지(단일 출력 모델 대비)
                    val = pred[0]; fail = abs(val) > self.threshold
                    status = "[FAULT]" if fail else "[NORMAL]"
                    color = 'white'
                    bgcolor = self.colors['danger'] if fail else self.colors['success']
                    border_color = '#dc3545' if fail else '#28a745'
                    label = f"100s DRIFT\n{val:.2f}°\n{status}"
            ax.text(0.5, 0.95, label, transform=ax.transAxes, ha='center', va='top',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=bgcolor,
                              edgecolor=border_color, linewidth=2, alpha=0.95),
                    color=color, fontfamily=self.font_family)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.state('zoomed')
    except Exception:
        root.attributes('-zoomed', True)
    app = IMUGUI(root)
    root.mainloop()
