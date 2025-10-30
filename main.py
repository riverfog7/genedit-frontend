"""
GenEdit v2 — 로컬 PySide6 UI (완전 주석 버전)
- 서버(백엔드) 엔드포인트는 환경변수 BACKEND 로 오버라이드 가능
- 주요 플로우: Detect -> Segment(포인트/박스/결합) -> Inpaint
- PySide6, requests, Pillow 필요

사용법(로컬):
  (Windows PowerShell 예)
  > $env:BACKEND = "http://127.0.0.1:8000"   # 또는 8008 등, SSH 포워딩 포트
  > python genedit_client_v2.py
"""

from __future__ import annotations  # 미래형 타입힌트 활성화 (Forward reference 등)

# 표준 라이브러리
import os  # 환경변수 읽기(BACKEND)
import sys  # 앱 종료, argv 등
from dataclasses import dataclass  # 단순 데이터 컨테이너 정의용
from typing import List, Optional, Tuple  # 타입 주석

import dotenv
# 서드파티 라이브러리
from PIL import Image, ImageQt  # 이미지 변환/처리, Qt 변환
# Qt 관련 (PySide6)
from PyQt6.QtCore import Qt, QRectF, QPointF  # 좌표/사각형 등 기본 타입
from PyQt6.QtGui import QPixmap, QImage, QPen, QColor, QPainter  # 그림 설정 및 페인터
from PyQt6.QtWidgets import (  # 다양한 위젯
    QApplication, QWidget, QSplitter, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QTextEdit, QFileDialog, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QListWidget, QListWidgetItem,
    QMessageBox, QCheckBox, QSpinBox
)

from client import DetectionClient, ImageGenerationClient, SegmentationClient, DetectorOutput

dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv())

BASE_URL=os.getenv("BACKEND_BASE_URL")
VLLM_BASE_URL=os.path.join(BASE_URL, "vllm/v1")
IMAGE_BASE_URL=os.path.join(BASE_URL, "image")

detection_client = DetectionClient(IMAGE_BASE_URL)
segmentation_client = SegmentationClient(IMAGE_BASE_URL)
generation_client = ImageGenerationClient(IMAGE_BASE_URL)

# ----------------------------- 캔버스 뷰 ---------------------------------------
class ImageCanvas(QGraphicsView):  # 이미지와 오버레이(박스/마스크/포인트)를 그리는 뷰
    def __init__(self, parent: Optional[QWidget] = None):  # 생성자
        super().__init__(parent)  # 부모 초기화
        # 안티앨리어싱 및 픽스맵 스무딩 렌더 힌트 설정 (QPainter 사용)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)  # 줌 기준점
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)  # 드래그로 이동
        self.scene = QGraphicsScene(self)  # 장면(Scene) 생성
        self.setScene(self.scene)  # 뷰에 장면 연결
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None  # 원본 이미지 아이템
        self.image: Optional[Image.Image] = None  # PIL 이미지 캐시
        self.drawing = False  # 드래그 중 사각형 그리기 여부
        self.rect_item: Optional[QGraphicsRectItem] = None  # 현재 드로잉 사각형
        self.rect_start = QPointF(0, 0)  # 드래그 시작점
        self.rect_end = QPointF(0, 0)  # 드래그 끝점
        self.overlay_items: List[QGraphicsRectItem] = []  # 탐지 오버레이 아이템 목록
        self.selected_box_item: Optional[QGraphicsRectItem] = None  # 선택 강조 아이템
        self.pos_points: List[Tuple[int, int]] = []  # 양성 포인트(Alt 미사용)
        self.neg_points: List[Tuple[int, int]] = []  # 음성 포인트(Alt 사용)
        self.mask_pixmap_item: Optional[QGraphicsPixmapItem] = None  # 마스크/결과 오버레이

    def load_image(self, path: str) -> None:  # 이미지 파일 로드
        self.image = Image.open(path).convert("RGBA")  # RGBA로 통일
        qimg = ImageQt.ImageQt(self.image)  # PIL→Qt 이미지 변환
        pix = QPixmap.fromImage(QImage(qimg))  # QPixmap 생성
        self.scene.clear()  # 장면 초기화
        self.pixmap_item = QGraphicsPixmapItem(pix)  # 픽스맵 아이템 생성
        self.scene.addItem(self.pixmap_item)  # 장면에 추가
        self.setSceneRect(QRectF(0, 0, pix.width(), pix.height()))  # 스크롤/뷰 경계 설정
        self.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)  # 보기 영역 맞춤
        self.rect_item = None  # 드래그 사각형 초기화
        self.clear_detections()  # 탐지 박스 초기화
        self.clear_points()  # 포인트 초기화
        self.clear_mask_overlay()  # 마스크 오버레이 초기화

    def clear_detections(self) -> None:  # 탐지 박스/선택 강조 제거
        for it in self.overlay_items:
            self.scene.removeItem(it)
        self.overlay_items = []
        if self.selected_box_item:
            self.scene.removeItem(self.selected_box_item)
            self.selected_box_item = None

    def clear_points(self) -> None:  # 클릭 포인트 초기화
        self.pos_points = []
        self.neg_points = []

    def clear_mask_overlay(self) -> None:  # 마스크/결과 오버레이 제거
        if self.mask_pixmap_item:
            self.scene.removeItem(self.mask_pixmap_item)
            self.mask_pixmap_item = None

    def draw_detections(self, detector_output: DetectorOutput, select_index: Optional[int] = None) -> None:  # 탐지 결과 그리기
        self.clear_detections()
        if self.image is None:
            return
        # Flatten all detections into a list of (box, score, label) tuples
        all_detections = []
        for det_result in detector_output.detections:
            for box, score, label in zip(det_result.boxes, det_result.scores, det_result.labels):
                all_detections.append((box, score, label))

        # Draw all boxes with blue solid lines
        for box, score, label in all_detections:
            x1, y1, x2, y2 = box
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            item = QGraphicsRectItem(rect)
            item.setPen(QPen(QColor(0, 120, 255), 2, Qt.PenStyle.SolidLine))
            item.setZValue(10)
            self.scene.addItem(item)
            self.overlay_items.append(item)

        # Draw selected box with yellow dashed line
        if select_index is not None and 0 <= select_index < len(all_detections):
            box, score, label = all_detections[select_index]
            x1, y1, x2, y2 = box
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            sel = QGraphicsRectItem(rect)
            sel.setPen(QPen(QColor(255, 210, 0), 3, Qt.PenStyle.DashLine))
            sel.setZValue(11)
            self.scene.addItem(sel)
            self.selected_box_item = sel

    def set_mask_overlay(self, pil_img: Image.Image) -> None:  # 마스크/결과 이미지를 오버레이로 표시
        if self.image is None:
            return
        w, h = self.image.size  # 원본 크기
        img = pil_img.convert("RGBA").resize((w, h))  # 오버레이 크기 맞추기
        qimg = ImageQt.ImageQt(img)  # PIL→Qt 변환
        pix = QPixmap.fromImage(QImage(qimg))  # 픽스맵 생성
        if self.mask_pixmap_item is None:  # 처음 추가하는 경우
            self.mask_pixmap_item = QGraphicsPixmapItem(pix)
            self.mask_pixmap_item.setZValue(6)  # 원본 위, 박스 아래 등 적절한 레이어
            self.scene.addItem(self.mask_pixmap_item)
        else:  # 기존 오버레이 갱신
            self.mask_pixmap_item.setPixmap(pix)

    def mousePressEvent(self, event):  # 마우스 눌림 이벤트
        if event.button() == Qt.MouseButton.LeftButton and self.pixmap_item is not None:
            modifiers = event.modifiers()  # 수정키 상태
            scene_pos = self.mapToScene(event.pos())  # 클릭 위치(씬 좌표)
            if modifiers & Qt.KeyboardModifier.AltModifier:  # Alt+좌클릭 → 음성 포인트
                self.neg_points.append((int(scene_pos.x()), int(scene_pos.y())))
                # 음성 포인트 클릭일 땐 사각형 드로잉은 시작하지 않도록 즉시 반환
                super().mousePressEvent(event)
                return
            else:  # 일반 좌클릭 → 양성 포인트 추가 & 드래그 사각형 시작
                self.pos_points.append((int(scene_pos.x()), int(scene_pos.y())))
                self.drawing = True
                self.rect_start = scene_pos
                if self.rect_item is None:
                    self.rect_item = QGraphicsRectItem()
                    self.rect_item.setZValue(12)
                    self.rect_item.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))
                    self.scene.addItem(self.rect_item)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # 마우스 이동(드래그 중 사각형 갱신)
        if self.drawing and self.rect_item is not None:
            self.rect_end = self.mapToScene(event.pos())
            rect = QRectF(self.rect_start, self.rect_end).normalized()
            self.rect_item.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # 마우스 떼기(드래그 종료)
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            self.rect_end = self.mapToScene(event.pos())
            if self.rect_item is not None:
                rect = QRectF(self.rect_start, self.rect_end).normalized()
                self.rect_item.setRect(rect)
        super().mouseReleaseEvent(event)

    def current_bbox(self) -> Optional[List[int]]:  # 현재 사용자가 그린 박스를 반환
        if self.rect_item is None:
            return None
        rect = self.rect_item.rect()
        if rect.width() < 5 or rect.height() < 5:  # 너무 작은 박스는 무시
            return None
        return [int(rect.x()), int(rect.y()),
                int(rect.x() + rect.width()), int(rect.y() + rect.height())]

# ----------------------------- 우측 채팅 패널 ----------------------------------
class RightChatPanel(QWidget):  # 간단 채팅 UI (백엔드에 채팅 라우트 없으면 안내 문구)
    def __init__(self, parent: Optional[QWidget] = None):  # 생성자
        super().__init__(parent)
        layout = QVBoxLayout(self)
        title = QLabel("<b>LLM 채팅</b>")
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.input = QLineEdit(); self.input.setPlaceholderText("명령/질문 입력…")
        self.btn_send = QPushButton("Send"); self.btn_send.clicked.connect(self.on_send)
        row = QHBoxLayout(); row.addWidget(self.input, 1); row.addWidget(self.btn_send)
        layout.addWidget(title); layout.addWidget(self.log, 1); layout.addLayout(row)

    def append(self, who: str, text: str):  # 로그에 라인 추가
        self.log.append(f"<b>{who}:</b> {text}")

    def on_send(self):  # 전송 핸들러
        txt = self.input.text().strip()
        if not txt:
            return
        self.append("You", txt)
        self.input.clear()
        reply = "채팅 기능은 아직 구현되지 않았습니다. vLLM 엔드포인트가 필요합니다."
        self.append("Assistant", reply)

# ----------------------------- 좌측 에디터 패널 --------------------------------
class LeftEditorPanel(QWidget):  # Detect → Segment → Inpaint 파이프라인 UI
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        title = QLabel("<b>이미지 편집 · Detect → Segment → Inpaint</b>")
        title.setStyleSheet("font-size:16px;")
        layout.addWidget(title)

        # (1) 이미지 열기/저장
        r1 = QHBoxLayout()
        self.btn_load = QPushButton("이미지 열기")
        self.btn_save = QPushButton("결과 저장"); self.btn_save.setEnabled(False)
        r1.addWidget(self.btn_load); r1.addWidget(self.btn_save); r1.addStretch(1)
        layout.addLayout(r1)

        # (2) Detect 설정
        r2 = QHBoxLayout()
        self.txt_detect = QLineEdit(); self.txt_detect.setPlaceholderText("세미콜론(;) 구분: cat; red cup")
        self.spn_thresh = QSpinBox(); self.spn_thresh.setRange(1, 99); self.spn_thresh.setValue(20)
        self.lbl_thresh = QLabel("threshold %")
        self.btn_detect = QPushButton("Detect")
        r2.addWidget(self.txt_detect, 1); r2.addWidget(self.lbl_thresh); r2.addWidget(self.spn_thresh); r2.addWidget(self.btn_detect)
        layout.addLayout(r2)

        # (2-1) Detect 결과 목록
        self.list_detect = QListWidget(); layout.addWidget(self.list_detect, 1)

        # (3) Segment 설정/버튼
        r3 = QHBoxLayout()
        self.chk_use_selected_bbox = QCheckBox("선택 bbox 사용")
        self.btn_clear_points = QPushButton("포인트 초기화")
        self.btn_segment = QPushButton("Segment (box+points)")
        r3.addWidget(self.chk_use_selected_bbox); r3.addWidget(self.btn_clear_points); r3.addWidget(self.btn_segment)
        layout.addLayout(r3)

        # (4) Inpaint 설정/버튼
        r4 = QHBoxLayout()
        self.txt_prompt = QLineEdit(); self.txt_prompt.setPlaceholderText("인페인트 프롬프트")
        self.spn_steps = QSpinBox(); self.spn_steps.setRange(1, 200); self.spn_steps.setValue(30)
        self.lbl_steps = QLabel("steps")
        self.btn_inpaint = QPushButton("Inpaint")
        r4.addWidget(self.txt_prompt, 1); r4.addWidget(self.lbl_steps); r4.addWidget(self.spn_steps); r4.addWidget(self.btn_inpaint)
        layout.addLayout(r4)

        # (5) 캔버스/상태
        self.canvas = ImageCanvas(); layout.addWidget(self.canvas, 3)
        self.status = QLabel("status: idle"); layout.addWidget(self.status)

        # 상태 변수들
        self.image_path: Optional[str] = None
        self.detector_output: Optional[DetectorOutput] = None
        self.selected_det_index: Optional[int] = None
        self.last_mask: Optional[Image.Image] = None
        self.last_result: Optional[Image.Image] = None

        # 시그널 연결
        self.btn_load.clicked.connect(self.load_image)
        self.btn_save.clicked.connect(self.save_result)
        self.btn_detect.clicked.connect(self.detect)
        self.list_detect.currentRowChanged.connect(self.on_select_detection)
        self.btn_clear_points.clicked.connect(self.on_clear_points)
        self.btn_segment.clicked.connect(self.segment)
        self.btn_inpaint.clicked.connect(self.inpaint)

    def set_status(self, text: str) -> None:  # 상태 텍스트 갱신 + UI 즉시 반영
        self.status.setText(text); QApplication.processEvents()

    def load_image(self) -> None:  # 이미지 열기 다이얼로그
        path, _ = QFileDialog.getOpenFileName(self, "이미지 선택", filter="Images (*.png *.jpg *.jpeg *.webp)")
        if not path:
            return
        self.image_path = path
        self.canvas.load_image(path)
        self.detector_output = None
        self.list_detect.clear()
        self.selected_det_index = None
        self.canvas.clear_mask_overlay(); self.last_mask = None; self.last_result = None
        self.btn_save.setEnabled(False)
        self.set_status(f"loaded: {path}")

    def save_result(self) -> None:  # 결과 이미지 저장
        if self.last_result is None:
            QMessageBox.information(self, "알림", "저장할 결과가 없습니다.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "결과 저장", filter="PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if not path:
            return
        fmt = "PNG" if path.lower().endswith(".png") else "JPEG"
        self.last_result.save(path, format=fmt)
        self.set_status(f"saved: {path}")

    def detect(self) -> None:  # 객체 탐지 API 호출
        if self.image_path is None:
            QMessageBox.warning(self, "경고", "먼저 이미지를 열어주세요.")
            return
        raw = self.txt_detect.text().strip()
        if not raw:
            QMessageBox.warning(self, "경고", "탐지할 객체를 입력하세요.")
            return
        labels = [s.strip() for s in raw.replace("\n", ";").split(";") if s.strip()]
        thr = self.spn_thresh.value() / 100.0
        try:
            self.set_status("detect: 요청 중...")
            detector_output = detection_client.detect(self.image_path, labels, thr)
            self.detector_output = detector_output
            self.list_detect.clear()

            # Flatten all detections for display
            for det_result in detector_output.detections:
                for box, score, label in zip(det_result.boxes, det_result.scores, det_result.labels):
                    x1, y1, x2, y2 = box
                    self.list_detect.addItem(QListWidgetItem(f"{label}  score={score:.2f}  box=({int(x1)},{int(y1)},{int(x2)},{int(y2)})"))

            self.canvas.draw_detections(detector_output, None)

            # Count total detections
            total_count = sum(len(det_result.boxes) for det_result in detector_output.detections)
            self.set_status(f"detect: {total_count}개")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"탐지 실패: {e}")
            self.set_status(f"detect: 오류 - {e}")

    def on_select_detection(self, row: int) -> None:  # 리스트 선택 변경 시 강조표시 갱신
        self.selected_det_index = row if row >= 0 else None
        if self.detector_output:
            self.canvas.draw_detections(self.detector_output, self.selected_det_index)

    def on_clear_points(self) -> None:  # 포인트 초기화 버튼
        self.canvas.clear_points()
        self.set_status("points cleared")

    def segment(self) -> None:  # 세그먼트(결합) API 호출
        if self.image_path is None:
            QMessageBox.warning(self, "경고", "먼저 이미지를 열어주세요.")
            return
        # 우선순위: 체크되었고 선택된 탐지 박스가 있으면 그걸 사용, 아니면 캔버스 드로잉 박스
        bbox: Optional[List[int]] = None
        if self.chk_use_selected_bbox.isChecked() and self.selected_det_index is not None:
            # Flatten detections to match list display
            all_detections = []
            if self.detector_output:
                for det_result in self.detector_output.detections:
                    for box, score, label in zip(det_result.boxes, det_result.scores, det_result.labels):
                        all_detections.append((box, score, label))

            if 0 <= self.selected_det_index < len(all_detections):
                box, _, _ = all_detections[self.selected_det_index]
                bbox = [int(x) for x in box]
        else:
            bbox = self.canvas.current_bbox()
        # 포인트/라벨 구성 (positive=1, negative=0)
        points: List[List[int]] = []
        labels: List[int] = []
        for x, y in self.canvas.pos_points:
            points.append([x, y]); labels.append(1)
        for x, y in self.canvas.neg_points:
            points.append([x, y]); labels.append(0)
        try:
            self.set_status("segment: 요청 중...")
            result = segmentation_client.segment_combined(
                self.image_path,
                points=points if points else None,
                labels=labels if labels else None,
                box=bbox
            )
            # Extract first mask from ZIP
            masks = result.extract_masks()
            if masks:
                self.last_mask = masks[0]
                self.canvas.set_mask_overlay(masks[0])
                self.set_status("segment: 완료")
            else:
                raise ValueError("No masks returned")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"세그먼트 실패: {e}")
            self.set_status(f"segment: 오류 - {e}")

    def inpaint(self) -> None:  # 인페인트 API 호출
        if self.image_path is None or self.last_mask is None:
            QMessageBox.warning(self, "경고", "이미지와 세그먼트 마스크가 필요합니다.")
            return
        prompt = self.txt_prompt.text().strip() or "remove object"
        steps = int(self.spn_steps.value())
        try:
            self.set_status("inpaint: 요청 중... (시간 소요)")
            result_img = generation_client.inpaint(
                control_image=self.image_path,
                control_mask=self.last_mask,
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=steps,
                true_cfg_scale=4.0,
                controlnet_conditioning_scale=1.0,
                seed=None
            )
            self.last_result = result_img
            # 결과를 즉시 오버레이로 보여줌 (원본 위에 결과가 보이도록)
            self.canvas.set_mask_overlay(result_img)
            self.btn_save.setEnabled(True)
            self.set_status("inpaint: 완료")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"인페인트 실패: {e}")
            self.set_status(f"inpaint: 오류 - {e}")

# ----------------------------- 메인 윈도우 --------------------------------------
class MainWindow(QWidget):  # 메인 프레임(좌: 에디터, 우: 채팅)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GenEdit v2 — 완전 주석 클라이언트")
        self.resize(1400, 900)

        splitter = QSplitter(Qt.Orientation.Horizontal)  # 좌/우 분할기
        self.left = LeftEditorPanel(); self.right = RightChatPanel()
        splitter.addWidget(self.left); splitter.addWidget(self.right)
        splitter.setSizes([1000, 400])  # 초기 너비 비율

        bar = QHBoxLayout()  # 상단 바: 헬스체크
        self.btn_health = QPushButton("헬스체크")
        self.lbl_health = QLabel("health: unknown")
        bar.addWidget(self.btn_health); bar.addWidget(self.lbl_health); bar.addStretch(1)

        root = QVBoxLayout(self)  # 루트 레이아웃 배치
        root.addLayout(bar); root.addWidget(splitter, 1)

        self.btn_health.clicked.connect(self.check_health)  # 버튼 연결

    def check_health(self):  # /health 호출하여 상태 갱신
        try:
            h1 = generation_client.health_check()
            h2 = detection_client.health_check()
            h3 = segmentation_client.health_check()
            self.lbl_health.setText(f"health: gen={h1.get('status')} detect={h2.get('status')} segment={h3.get('status')}")
        except Exception as e:
            self.lbl_health.setText(f"health error: {e}")

# ----------------------------- 진입점 ------------------------------------------
if __name__ == "__main__":  # 스크립트 직접 실행 시
    app = QApplication(sys.argv)  # Qt 애플리케이션 생성
    w = MainWindow()  # 메인 윈도우 생성
    w.show()  # 창 띄우기
    sys.exit(app.exec())  # 이벤트 루프 시작 및 종료 코드 반환
