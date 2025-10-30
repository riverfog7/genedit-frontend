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
import io  # 바이트 스트림 처리용(마스크/이미지 직렬화)
import json  # 폼의 'data' 필드에 넣을 JSON 직렬화
import os  # 환경변수 읽기(BACKEND)
import sys  # 앱 종료, argv 등
from dataclasses import dataclass  # 단순 데이터 컨테이너 정의용
from typing import List, Optional, Tuple  # 타입 주석

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from client import DetectionClient, SegmentationClient, ImageGenerationClient
from client.utils import draw_detections

# 서드파티 라이브러리
import requests  # HTTP 클라이언트
from PIL import Image, ImageQt  # 이미지 변환/처리, Qt 변환

# Qt 관련 (PySide6)
from PySide6.QtCore import Qt, QRectF, QPointF  # 좌표/사각형 등 기본 타입
from PySide6.QtGui import QPixmap, QImage, QPen, QColor, QPainter  # 그림 설정 및 페인터
from PySide6.QtWidgets import (  # 다양한 위젯
    QApplication, QWidget, QSplitter, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QTextEdit, QFileDialog, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QListWidget, QListWidgetItem,
    QMessageBox, QCheckBox, QSpinBox
)

# ----------------------------- 엔드포인트 설정 --------------------------------
BACKEND = os.environ.get("BACKEND", "http://localhost:8000")  # 기본값은 8000; 필요시 환경변수로 변경
EP_HEALTH = f"{BACKEND}/health"  # 헬스체크
EP_DETECT = f"{BACKEND}/detect"  # 객체 탐지
EP_SEG_POINT = f"{BACKEND}/segment/point"  # 포인트 세그먼트
EP_SEG_BOX = f"{BACKEND}/segment/box"  # 박스 세그먼트
EP_SEG_COMBINED = f"{BACKEND}/segment/combined"  # 결합 세그먼트
EP_INPAINT = f"{BACKEND}/generate/inpaint"  # 인페인트(이미지 수정)
EP_GENERATE = f"{BACKEND}/generate/generate"  # 텍스트→이미지 생성(선택)
EP_CHAT = f"{BACKEND}/v1/chat/completions"  # (레포에 없을 수 있음) 채팅 라우트
CHAT_OPENAI_COMPAT = False  # 현재 백엔드에 채팅 라우트가 없으므로 False 유지
OPENAI_MODEL_NAME = "model"  # 채팅 사용시 모델명(참고용)

# ----------------------------- 데이터 모델 -------------------------------------
@dataclass
class BBox:  # 바운딩 박스 좌표 컨테이너
    x1: int  # 좌상단 x
    y1: int  # 좌상단 y
    x2: int  # 우하단 x
    y2: int  # 우하단 y
    def to_list(self) -> List[int]:  # API 전송용 리스트 변환
        return [int(self.x1), int(self.y1), int(self.x2), int(self.y2)]

@dataclass
class Detection:  # 탐지 결과 항목
    box: BBox      # 박스
    score: float   # 신뢰도
    label: str     # 어떤 텍스트 프롬프트에 해당하는지

# ----------------------------- 캔버스 뷰 ---------------------------------------
class ImageCanvas(QGraphicsView):  # 이미지와 오버레이(박스/마스크/포인트)를 그리는 뷰
    def __init__(self, parent: Optional[QWidget] = None):  # 생성자
        super().__init__(parent)  # 부모 초기화
        # 안티앨리어싱 및 픽스맵 스무딩 렌더 힌트 설정 (QPainter 사용)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)  # 줌 기준점
        self.setDragMode(QGraphicsView.ScrollHandDrag)  # 드래그로 이동
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
        self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)  # 보기 영역 맞춤
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

    def draw_detections(self, dets: List[Detection], select_index: Optional[int] = None) -> None:  # 탐지 결과 그리기
        self.clear_detections()
        if self.image is None:
            return
        for det in dets:  # 모든 박스에 파란 실선 오버레이
            rect = QRectF(det.box.x1, det.box.y1, det.box.x2 - det.box.x1, det.box.y2 - det.box.y1)
            item = QGraphicsRectItem(rect)
            item.setPen(QPen(QColor(0, 120, 255), 2, Qt.SolidLine))
            item.setZValue(10)
            self.scene.addItem(item)
            self.overlay_items.append(item)
        if select_index is not None and 0 <= select_index < len(dets):  # 선택 박스는 노랑 점선
            b = dets[select_index].box
            rect = QRectF(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1)
            sel = QGraphicsRectItem(rect)
            sel.setPen(QPen(QColor(255, 210, 0), 3, Qt.DashLine))
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
        if event.button() == Qt.LeftButton and self.pixmap_item is not None:
            modifiers = event.modifiers()  # 수정키 상태
            scene_pos = self.mapToScene(event.pos())  # 클릭 위치(씬 좌표)
            if modifiers & Qt.AltModifier:  # Alt+좌클릭 → 음성 포인트
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
                    self.rect_item.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
                    self.scene.addItem(self.rect_item)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # 마우스 이동(드래그 중 사각형 갱신)
        if self.drawing and self.rect_item is not None:
            self.rect_end = self.mapToScene(event.pos())
            rect = QRectF(self.rect_start, self.rect_end).normalized()
            self.rect_item.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # 마우스 떼기(드래그 종료)
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.rect_end = self.mapToScene(event.pos())
            if self.rect_item is not None:
                rect = QRectF(self.rect_start, self.rect_end).normalized()
                self.rect_item.setRect(rect)
        super().mouseReleaseEvent(event)

    def current_bbox(self) -> Optional[BBox]:  # 현재 드래그 사각형을 BBox로 반환
        if self.rect_item is None or self.image is None:
            return None
        rect: QRectF = self.rect_item.rect()
        # 이미지 경계 안으로 좌표 클램프
        x1 = max(0, min(int(rect.left()),   self.image.width  - 1))
        y1 = max(0, min(int(rect.top()),    self.image.height - 1))
        x2 = max(0, min(int(rect.right()),  self.image.width  - 1))
        y2 = max(0, min(int(rect.bottom()), self.image.height - 1))
        if x2 <= x1 or y2 <= y1:  # 잘못된 사각형이면 None
            return None
        return BBox(x1, y1, x2, y2)

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
        try:
            reply = send_chat_message(txt)
        except Exception as e:
            reply = f"(error) {e}"
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
        self.detections: List[Detection] = []
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
        self.detections = []
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
            dets = call_detect_api(self.image_path, labels, thr)
            self.detections = dets
            self.list_detect.clear()
            for d in dets:
                self.list_detect.addItem(QListWidgetItem(f"{d.label}  score={d.score:.2f}  box=({d.box.x1},{d.box.y1},{d.box.x2},{d.box.y2})"))
            self.canvas.draw_detections(dets, None)
            self.set_status(f"detect: {len(dets)}개")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"탐지 실패: {e}")
            self.set_status(f"detect: 오류 - {e}")

    def on_select_detection(self, row: int) -> None:  # 리스트 선택 변경 시 강조표시 갱신
        self.selected_det_index = row if row >= 0 else None
        self.canvas.draw_detections(self.detections, self.selected_det_index)

    def on_clear_points(self) -> None:  # 포인트 초기화 버튼
        self.canvas.clear_points()
        self.set_status("points cleared")

    def segment(self) -> None:  # 세그먼트(결합) API 호출
        if self.image_path is None:
            QMessageBox.warning(self, "경고", "먼저 이미지를 열어주세요.")
            return
        # 우선순위: 체크되었고 선택된 탐지 박스가 있으면 그걸 사용, 아니면 캔버스 드로잉 박스
        bbox: Optional[BBox] = None
        if self.chk_use_selected_bbox.isChecked() and self.selected_det_index is not None and 0 <= self.selected_det_index < len(self.detections):
            bbox = self.detections[self.selected_det_index].box
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
            mask_img = call_segment_combined_api(self.image_path, bbox, points if points else None, labels if labels else None)
            self.last_mask = mask_img
            self.canvas.set_mask_overlay(mask_img)
            self.set_status("segment: 완료")
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
            result_img = call_inpaint_api(
                control_image_path=self.image_path,
                control_mask_image=self.last_mask,
                prompt=prompt,
                num_inference_steps=steps,
                true_cfg_scale=4.0,
                negative_prompt="",
                controlnet_conditioning_scale=1.0,
                seed=None,
            )
            self.last_result = result_img
            # 결과를 즉시 오버레이로 보여줌 (원본 위에 결과가 보이도록)
            self.canvas.set_mask_overlay(result_img)
            self.btn_save.setEnabled(True)
            self.set_status("inpaint: 완료")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"인페인트 실패: {e}")
            self.set_status(f"inpaint: 오류 - {e}")

# ----------------------------- HTTP 헬퍼 ----------------------------------------

def parse_image_bytes_resp(resp: requests.Response) -> Image.Image:  # 이미지 바이트 응답을 PIL 이미지로 변환
    ctype = resp.headers.get("Content-Type", "")
    if "image" in ctype or resp.content:  # 이미지 응답이면
        try:
            return Image.open(io.BytesIO(resp.content)).convert("RGBA")
        except Exception as e:
            raise RuntimeError(f"이미지 디코드 실패: {e}")
    raise RuntimeError(f"알 수 없는 응답 형식(Content-Type={ctype})")


def call_detect_api(image_path: str, texts: List[str], threshold: float) -> List[Detection]:  # 객체 탐지 API 호출
    # 파일은 with 블록으로 열어 자동으로 닫히게 함
    with open(image_path, "rb") as f:
        files = {"image": f}  # 파일 필드명은 백엔드 스키마에 맞춤
        payload = {"text": texts, "threshold": threshold}  # 백엔드가 기대하는 구조
        data = {"data": json.dumps(payload)}  # multipart form 의 'data' 키에 JSON 문자열
        r = requests.post(EP_DETECT, files=files, data=data, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} {r.text[:200]}")
    try:
        js = r.json()  # JSON 파싱
        detections = js.get("detections", [])  # DetectorOutput.detections
        out: List[Detection] = []
        for det_block in detections:  # 각 블록은 boxes/scores/labels 를 가짐
            boxes = det_block.get("boxes", [])
            scores = det_block.get("scores", [])
            labels = det_block.get("labels", [])
            for b, s, lb in zip(boxes, scores, labels):  # 동일 길이 가정
                bb = BBox(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                out.append(Detection(bb, str(lb), float(s)))
        return out
    except Exception as e:
        raise RuntimeError(f"detect 응답 파싱 실패: {e}")


def call_segment_combined_api(image_path: str, bbox: Optional[BBox], points: Optional[List[List[int]]], labels: Optional[List[int]]) -> Image.Image:
    # 결합형 세그먼트 호출 (box/points 둘 다 또는 일부만)
    with open(image_path, "rb") as f:
        files = {"image": f}
        payload = {}
        if bbox is not None:
            # 서버가 'box' 키를 기대한다고 가정. 만약 'boxes'를 기대한다면 아래 한 줄을 교체:
            # payload["boxes"] = bbox.to_list()
            payload["box"] = bbox.to_list()
        if points is not None:
            payload["points"] = points
        if labels is not None:
            payload["labels"] = labels
        data = {"data": json.dumps(payload)}
        r = requests.post(EP_SEG_COMBINED, files=files, data=data, timeout=600)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} {r.text[:200]}")
    ctype = r.headers.get("Content-Type", "")
    # 서버가 ZIP(application/zip)으로 마스크/메타데이터를 반환하는 경우 지원
    if "application/zip" in ctype or r.content[:2] == b"PK":
        import zipfile
        buf = io.BytesIO(r.content)
        with zipfile.ZipFile(buf) as zf:
            png_names = [n for n in zf.namelist() if n.lower().endswith('.png')]
            if not png_names:
                raise RuntimeError("ZIP 응답에 PNG 마스크가 없습니다")
            raw = zf.read(png_names[0])
            return Image.open(io.BytesIO(raw)).convert("RGBA")
    # ZIP 이 아니면 일반 이미지로 파싱 시도
    return parse_image_bytes_resp(r)


def call_inpaint_api(
    control_image_path: str,
    control_mask_image: Image.Image,
    prompt: str,
    num_inference_steps: int = 30,
    true_cfg_scale: float = 4.0,
    negative_prompt: str = "",
    controlnet_conditioning_scale: float = 1.0,
    seed: Optional[int] = None,
) -> Image.Image:
    # 마스크 이미지를 PNG 바이트로 직렬화
    mask_buf = io.BytesIO(); control_mask_image.save(mask_buf, format="PNG"); mask_buf.seek(0)
    with open(control_image_path, "rb") as f:
        files = {
            "control_image": f,  # 원본 이미지 파일
            "control_mask": ("mask.png", mask_buf, "image/png"),  # (파일명, 바이트, MIME)
        }
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": true_cfg_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
        }
        if seed is not None:
            payload["seed"] = int(seed)
        data = {"data": json.dumps(payload)}
        r = requests.post(EP_INPAINT, files=files, data=data, timeout=1200)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} {r.text[:200]}")
    return parse_image_bytes_resp(r)


def send_chat_message(user_text: str) -> str:
    if not CHAT_OPENAI_COMPAT:
        return "(chat not available on this backend)"  # 미지원 시 안내
    payload = {
        "model": OPENAI_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for image editing."},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.2,
    }
    r = requests.post(EP_CHAT, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} {r.text[:200]}")
    js = r.json()
    try:
        return js["choices"][0]["message"]["content"]
    except Exception:
        return str(js)[:800]

# ----------------------------- 메인 윈도우 --------------------------------------
class MainWindow(QWidget):  # 메인 프레임(좌: 에디터, 우: 채팅)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GenEdit v2 — 완전 주석 클라이언트")
        self.resize(1400, 900)

        splitter = QSplitter(Qt.Horizontal)  # 좌/우 분할기
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
            r = requests.get(EP_HEALTH, timeout=5)
            self.lbl_health.setText(f"health: {r.status_code} {r.text[:60]}")
        except Exception as e:
            self.lbl_health.setText(f"health error: {e}")

# ----------------------------- 진입점 ------------------------------------------
if __name__ == "__main__":  # 스크립트 직접 실행 시
    app = QApplication(sys.argv)  # Qt 애플리케이션 생성
    w = MainWindow()  # 메인 윈도우 생성
    w.show()  # 창 띄우기
    sys.exit(app.exec())  # 이벤트 루프 시작 및 종료 코드 반환


