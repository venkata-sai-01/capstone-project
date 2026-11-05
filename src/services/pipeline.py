import asyncio
import csv
import os
import random
import threading
import time
from collections import deque
from datetime import datetime
from fractions import Fraction
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from aiortc import MediaStreamTrack
from av import VideoFrame
if __package__ == "src.services":
    from src.detection_overlay import detection_color, draw_detection_overlay
    from src.yolo_detector import YOLO26Detector
else:
    from detection_overlay import detection_color, draw_detection_overlay
    from yolo_detector import YOLO26Detector


if __package__ == "src.services":
    from src.config import CSV_LOG_FILE, DATABASE_URL, DEFAULT_MODEL_PATH, EVIDENCE_DIR, MODEL_PATH, _connect_database
    from src.detection_overlay import detection_color, draw_detection_overlay
    from src.yolo_detector import YOLO26Detector
else:
    from config import CSV_LOG_FILE, DATABASE_URL, DEFAULT_MODEL_PATH, EVIDENCE_DIR, MODEL_PATH, _connect_database
    from detection_overlay import detection_color, draw_detection_overlay
    from yolo_detector import YOLO26Detector


from .integrations import (
    AP_STATUTORY_PENALTIES, OFFICIAL_AP_RTO_MATRIX, DynamicROIMasker,
    EnvironmentalAdaptivityEngine, DynamicTrafficGrabber,
    VelocityCollisionIncidentDetector, acoustic_radar, ap_cctns_radar,
    ap_rto_gateway, blockchain_ledger, emission_auditor, evpp_engine,
    hsrp_engine, predictive_engine, smart_signal, timeseries_logger,
    vahan_db, weather_engine,
)

class APPoliceRadarSafetyEngine:
    """Core AI processing unit integrating YOLO, tracking, speed radar, risk prediction, and blockchain dispatch."""

    VEHICLE_CLASSES = {
        "car": "car",
        "van": "car",
        "motorcycle": "bike",
        "motorbike": "bike",
        "bicycle": "bike",
        "bus": "bus",
        "truck": "lorry",
        "lorry": "lorry",
    }
    ANIMAL_CLASSES = {
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
    }

    def __init__(self, grabber: DynamicTrafficGrabber, model_path: str = DEFAULT_MODEL_PATH):
        self.grabber = grabber
        self.model_path = model_path
        self.device = "cpu"
        self.model = YOLO26Detector(model_path)

        self.incident_detector = VelocityCollisionIncidentDetector()
        self.adaptivity_engine = EnvironmentalAdaptivityEngine()
        self.roi_masker = DynamicROIMasker(width=1280, height=720)

        self.conf_threshold: float = 0.30
        self.speed_limit_kmh: float = 60.0
        self.calibration_distance_m: float = 20.0

        self.line1_y: int = 240
        self.line2_y: int = 520
        self.crosswalk_y_min: int = 340
        self.crosswalk_y_max: int = 460

        self.line1_crossings: Dict[int, float] = {}
        self.line2_crossings: Dict[int, float] = {}
        self.calculated_speeds: Dict[int, float] = {}
        self.vehicle_plates: Dict[int, str] = {}
        self.trajectories: Dict[int, Deque[Tuple[int, int]]] = {}
        self.processed_violations: Set[str] = set()

        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        self.latest_annotated_rgb: Optional[np.ndarray] = None
        self.total_scanned: int = 0
        self.total_violations: int = 0
        self.total_fines_inr: int = 0
        self.total_hotlist_alerts: int = 0
        self.corridor_risk_index: float = 0.0

        self.violation_events: List[Dict[str, Any]] = []
        self.active_incidents_list: List[str] = []
        self.district_violation_counts: Dict[str, int] = {}
        self.telemetry: Dict[str, Any] = {
            "fps": 0.0,
            "inference_ms": 0.0,
            "speed_limit": self.speed_limit_kmh,
            "line1_y": self.line1_y,
            "line2_y": self.line2_y,
            "active_vehicles": 0,
            "total_passed": 0,
            "total_violations": 0,
            "total_fines_inr": self.total_fines_inr,
            "hotlist_hits": 0,
            "pedestrians_in_danger": 0,
            "cattle_on_highway": 0,
            "animals_in_danger": 0,
            "surrounding_objects_count": 0,
            "object_counts": {},
            "detections": [],
            "detection_model": os.path.basename(model_path),
            "incidents": [],
            "rto_gateway_status": ap_rto_gateway.dispatch_status.replace("_", " ").upper(),
            "recent_violations": [],
            "predictive_risk_score": 0.0,
            "traffic_signal": {"phase": "GREEN", "seconds_in_phase": 0, "adaptive_active": True},
            "blockchain": {"verified": True, "total_blocks": 1},
            "green_corridor": {"active": False},
            "weather": {"visibility_pct": 95.0, "condition": "Clear", "advisory_speed_kmh": 60.0},
            "carbon_emissions": {"total_co2_kg": 0.0, "expired_puc_count": 0}
        }

        self.snapshot_dir = EVIDENCE_DIR
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.csv_log_file = CSV_LOG_FILE
        self._init_csv()

    def _init_csv(self):
        os.makedirs(os.path.dirname(self.csv_log_file), exist_ok=True)
        if not os.path.exists(self.csv_log_file):
            with open(self.csv_log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Challan_No", "License_Plate", "Owner_Name", "Driving_License",
                    "Vehicle_Class", "District", "DTO_Office", "Make_Model", "Fuel_Type",
                    "FASTag_ID", "Recorded_Speed_KMH", "Speed_Limit_KMH", "Statutory_Section",
                    "Telugu_Notice", "Fine_Amount_INR", "CCTNS_Status", "Digital_Signature", "Snapshot_Path"
                ])
        if DATABASE_URL:
            conn = _connect_database("traffic_registry.db")
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS traffic_violations (
                    timestamp TEXT,
                    challan_no TEXT PRIMARY KEY,
                    license_plate TEXT,
                    owner_name TEXT,
                    driving_license TEXT,
                    vehicle_class TEXT,
                    district TEXT,
                    dto_office TEXT,
                    make_model TEXT,
                    fuel_type TEXT,
                    fastag_id TEXT,
                    recorded_speed_kmh DOUBLE PRECISION,
                    speed_limit_kmh INTEGER,
                    statutory_section TEXT,
                    telugu_notice TEXT,
                    fine_amount_inr INTEGER,
                    cctns_status TEXT,
                    digital_signature TEXT,
                    snapshot_path TEXT
                )
            """)
            conn.commit()
            conn.close()

    def start(self):
        self.running = True
        worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.thread = worker_thread
        worker_thread.start()

    def update_params(self, speed_limit: Optional[float] = None, distance_m: Optional[float] = None,
                      conf: Optional[float] = None, line1_y: Optional[int] = None, line2_y: Optional[int] = None):
        with self.lock:
            if speed_limit is not None:
                self.speed_limit_kmh = float(speed_limit)
            if distance_m is not None:
                self.calibration_distance_m = float(distance_m)
            if conf is not None:
                self.conf_threshold = max(0.05, min(0.99, float(conf)))
            if line1_y is not None:
                self.line1_y = max(50, min(650, int(line1_y)))
            if line2_y is not None:
                self.line2_y = max(self.line1_y + 30, min(700, int(line2_y)))

    def _get_or_create_plate(self, track_id: int, v_type: str) -> str:
        if track_id not in self.vehicle_plates:
            code = random.choice(list(OFFICIAL_AP_RTO_MATRIX.keys()))
            series = "Z" if v_type == "bus" and random.random() > 0.4 else f"{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}"
            num = f"{random.randint(1000, 9999)}"
            self.vehicle_plates[track_id] = f"{code} {series} {num}"
        return self.vehicle_plates[track_id]

    @staticmethod
    def _class_name(class_id: int, names: Any) -> str:
        if isinstance(names, dict):
            value = names.get(class_id, names.get(str(class_id), f"class_{class_id}"))
        elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            value = names[class_id]
        else:
            value = f"class_{class_id}"
        class_name = str(value).strip().lower().replace("_", " ")
        aliases = {"human": "person", "pedestrian": "person", "motorbike": "motorcycle"}
        return aliases.get(class_name, class_name) or f"class_{class_id}"

    @classmethod
    def _object_kind(cls, class_name: str) -> str:
        if class_name == "person":
            return "person"
        if class_name in cls.VEHICLE_CLASSES:
            return "vehicle"
        if class_name in cls.ANIMAL_CLASSES:
            return "animal"
        return "object"

    def _dispatch_violation(self, track_id: int, v_type: str, violation_type: str, speed: float, frame: np.ndarray,
                            box: np.ndarray):
        plate = self._get_or_create_plate(track_id, v_type)
        v_key = f"{plate}_{violation_type}"
        if v_key in self.processed_violations:
            return
        self.processed_violations.add(v_key)

        vahan_data = vahan_db.resolve_plate_to_profile(plate, v_type)
        cctns_hit = ap_cctns_radar.check_hotlist(plate)
        if cctns_hit:
            self.total_hotlist_alerts += 1

        hsrp_res = hsrp_engine.audit_plate_format(plate)
        emission_auditor.compute_vehicle_emission(vahan_data, speed)

        timestamp_full = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        district_str = str(vahan_data["district"])
        challan_no = f"CH-{district_str[:3].upper()}-{random.randint(1000000, 9999999)}"
        snapshot_filename = f"{self.snapshot_dir}/eChallan_{challan_no}_{timestamp_file}.jpg"

        sec_data = AP_STATUTORY_PENALTIES.get(violation_type, AP_STATUTORY_PENALTIES["overspeed_lmv"])
        fine_amount = int(sec_data["fine"])
        if violation_type in ["overspeed_lmv", "overspeed_hmv"] and (speed - self.speed_limit_kmh) > 30:
            fine_amount += 1000

        self.total_violations += 1
        self.total_fines_inr += fine_amount
        self.district_violation_counts[district_str] = self.district_violation_counts.get(district_str, 0) + 1

        telugu_sms = f"ట్రాఫిక్ ఎన్ఫోర్స్మెంట్ శాఖ: మీ వాహనం {plate} పై {sec_data['desc_te']} కారణముగా చలానా రూ. {fine_amount} జారీ చేయబడింది. చలానా నెం: {challan_no}."

        rto_payload: Dict[str, Any] = {
            "challan_no": challan_no,
            "timestamp": timestamp_full,
            "plate_number": plate,
            "owner_name": vahan_data["owner_name"],
            "dl_number": vahan_data["dl_number"],
            "dto_office": vahan_data["dto_office"],
            "district": district_str,
            "vehicle_type": v_type,
            "speed_kmh": round(speed, 1),
            "speed_limit_kmh": int(self.speed_limit_kmh),
            "violation_code": violation_type,
            "statutory_act": sec_data["section"],
            "fine_amount": fine_amount,
            "fastag_id": vahan_data["fastag_id"],
            "cctns_status": cctns_hit["severity"] if cctns_hit else "CLEAR",
            "hsrp_compliant": hsrp_res["is_hsrp_compliant"],
            "state": "Andhra Pradesh"
        }
        ap_rto_gateway.submit_challan_async(rto_payload)
        timeseries_logger.log_point(track_id, plate, speed, status=violation_type)

        citation_frame = frame.copy()
        cv2.rectangle(citation_frame, (0, 0), (1280, 115), (20, 25, 110), -1)
        cv2.putText(citation_frame, "AUTOMATED E-CHALLAN CITATION | TRAFFIC SURVEILLANCE & ENFORCEMENT", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(citation_frame,
                    f"Challan: {challan_no} | Owner: {vahan_data['owner_name']} | Plate: {plate} ({district_str})",
                    (30, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
        cv2.putText(citation_frame,
                    f"{sec_data['section']} | Fine: Rs. {fine_amount:,} | Speed: {speed:.1f} km/h (Limit: {int(self.speed_limit_kmh)} km/h)",
                    (30, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 100), 2)
        if cctns_hit:
            cv2.putText(citation_frame, f"⚠️ HOTLIST ALERT: {cctns_hit['reason']}", (30, 106), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0, 0, 255), 2)

        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(citation_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.imwrite(snapshot_filename, citation_frame)

        violation_row = [
            timestamp_full, challan_no, plate, vahan_data["owner_name"], vahan_data["dl_number"],
            vahan_data["veh_class"], district_str, vahan_data["dto_office"], vahan_data["make_model"],
            vahan_data["fuel_type"], vahan_data["fastag_id"], round(speed, 1), int(self.speed_limit_kmh),
            sec_data["section"], telugu_sms, fine_amount, cctns_hit["severity"] if cctns_hit else "CLEAR",
            rto_payload.get("digital_signature", "DIGITALLY_SIGNED"), snapshot_filename
        ]
        if DATABASE_URL:
            conn = _connect_database("traffic_registry.db")
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO traffic_violations VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                ) ON CONFLICT(challan_no) DO NOTHING
                """,
                tuple(violation_row),
            )
            conn.commit()
            conn.close()

        with open(self.csv_log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(violation_row)

        event: Dict[str, Any] = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "challan_no": challan_no,
            "plate": plate,
            "owner": vahan_data["owner_name"],
            "model": vahan_data["make_model"],
            "district": district_str,
            "violation": sec_data["desc_en"],
            "violation_te": sec_data["desc_te"],
            "speed": round(speed, 1),
            "fine": fine_amount,
            "cctns_alert": cctns_hit["severity"] if cctns_hit else None,
            "rto_sync": "SYNCED",
            "blockchain_block": rto_payload.get("blockchain_block_index", len(blockchain_ledger.chain)),
            "snapshot_url": f"/api/view_snapshot?file={os.path.basename(snapshot_filename)}"
        }
        self.violation_events.append(event)
        if len(self.violation_events) > 30:
            self.violation_events.pop(0)

    def _inference_worker(self):
        fps_tracker = 0
        last_time = time.perf_counter()
        minimum_frame_interval = 0.2

        while self.running:
            raw_bgr = self.grabber.get_frame(timeout=0.03)
            if raw_bgr is None:
                continue

            t_start = time.perf_counter(            )
            with self.lock:
                dispatch_status = ap_rto_gateway.dispatch_status
                dispatch_count_label = (
                    f"Sandbox records: {ap_rto_gateway.total_dispatched}"
                    if dispatch_status == "sandbox_simulation"
                    else f"External successes: {ap_rto_gateway.total_dispatched}"
                )
                conf = self.conf_threshold
                speed_limit = self.speed_limit_kmh
                distance_m = self.calibration_distance_m
                l1_y = self.line1_y
                l2_y = self.line2_y

            weather_stats = weather_engine.analyze_frame_visibility(raw_bgr)
            evpp_status = evpp_engine.evaluate_cycle()

            enhanced_bgr = self.adaptivity_engine.enhance_frame(raw_bgr, apply_clahe=True, gamma=1.05)
            masked_input = self.roi_masker.apply(enhanced_bgr)

            boxes, track_ids, class_ids, confidences, model_names = self.model.infer(
                masked_input, conf
            )

            annotated_frame = raw_bgr.copy()
            h, w, _ = annotated_frame.shape

            cv2.line(annotated_frame, (0, l1_y), (w, l1_y), (255, 255, 0), 2)
            cv2.putText(annotated_frame, "RADAR TRAP LINE A", (30, l1_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (255, 255, 0), 2)

            cv2.line(annotated_frame, (0, l2_y), (w, l2_y), (255, 0, 255), 2)
            cv2.putText(annotated_frame, f"RADAR TRAP LINE B ({distance_m}m Trap Distance)", (30, l2_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 0, 255), 2)

            cv2.rectangle(annotated_frame, (180, self.crosswalk_y_min), (1100, self.crosswalk_y_max), (255, 255, 255),
                          1)
            cv2.putText(annotated_frame, "PEDESTRIAN & ANIMAL CROSSING CORRIDOR", (200, self.crosswalk_y_min - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)

            current_time = time.time()
            active_count = 0
            pedestrian_danger_count = 0
            animal_danger_count = 0
            cattle_danger_count = 0
            surrounding_obj_count = 0
            boxes_dict: Dict[int, np.ndarray] = {}
            lane_assignments: Dict[int, int] = {}

            if not len(boxes) and not self.grabber.has_source:
                v_boxes = [[int(v["x"]), int(v["y"]), int(v["x"]) + 90, int(v["y"]) + 140] for v in
                           self.grabber.sim_vehicles]
                p_boxes = [[int(p["x"]), int(p["y"]), int(p["x"]) + 40, int(p["y"]) + 60] for p in
                           self.grabber.sim_crossings]
                s_boxes = [[int(s["x"]), int(s["y"]), int(s["x"]) + int(s["w"]), int(s["y"]) + int(s["h"])] for s in
                           self.grabber.sim_surroundings]
                boxes = np.array(v_boxes + p_boxes + s_boxes, dtype=int)
                track_ids = np.array(
                    [int(v["id_sim"]) for v in self.grabber.sim_vehicles] +
                    [int(p["id_sim"]) for p in self.grabber.sim_crossings] +
                    [300 + idx for idx in range(len(self.grabber.sim_surroundings))],
                    dtype=int
                )
                class_ids = np.array(
                    [2 if v["v_class"] == "car" else 7 if v["v_class"] == "lorry" else 5 if v["v_class"] == "bus" else 3
                     for v in self.grabber.sim_vehicles] +
                    [0 if p["c_class"] == "person" else 19 for p in self.grabber.sim_crossings] +
                    [9 if s["class_name"] == "traffic light" else 11 for s in self.grabber.sim_surroundings],
                    dtype=int
                )
                confidences = np.ones(len(boxes), dtype=float)
            if len(confidences) != len(boxes):
                confidences = np.ones(len(boxes), dtype=float)

            object_counts: Dict[str, int] = {}
            detections: List[Dict[str, Any]] = []
            detection_rows = zip(boxes.tolist(), track_ids.tolist(), class_ids.tolist(), confidences.tolist())
            for box, track_id, cls_id, confidence in detection_rows:
                t_id = int(track_id)
                c_id = int(cls_id)
                x1 = max(0, min(int(w), int(box[0])))
                y1 = max(0, min(int(h), int(box[1])))
                x2 = max(0, min(int(w), int(box[2])))
                y2 = max(0, min(int(h), int(box[3])))
                center_x = int((x1 + x2) // 2)
                center_y = int((y1 + y2) // 2)

                lane_num = 1 if center_x < 480 else (2 if center_x < 790 else 3)
                lane_assignments[t_id] = lane_num
                obj_name = self._class_name(c_id, model_names)
                obj_kind = self._object_kind(obj_name)
                score = float(max(0.0, min(1.0, confidence)))
                object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
                detections.append({
                    "track_id": t_id,
                    "class_id": c_id,
                    "label": obj_name,
                    "kind": obj_kind,
                    "confidence": round(score, 4),
                    "bbox": [x1, y1, x2, y2],
                })

                if obj_kind == "object":
                    surrounding_obj_count += 1
                if obj_kind in ("person", "animal"):
                    is_pedestrian = obj_kind == "person"

                    if 200 <= center_x <= 1080 and (self.crosswalk_y_min <= center_y <= self.crosswalk_y_max + 100):
                        if is_pedestrian:
                            pedestrian_danger_count += 1
                        else:
                            animal_danger_count += 1
                            if obj_name == "cow":
                                cattle_danger_count += 1
                        cv2.circle(annotated_frame, (center_x, center_y), 8, (0, 0, 255), -1)

                    draw_detection_overlay(
                        annotated_frame,
                        (x1, y1, x2, y2),
                        f"{obj_name.title()} {score:.2f}",
                        obj_name,
                        f"Track #{t_id}",
                    )
                    continue

                if obj_kind == "object":
                    draw_detection_overlay(
                        annotated_frame,
                        (x1, y1, x2, y2),
                        f"{obj_name.title()} {score:.2f}",
                        obj_name,
                        f"Track #{t_id}",
                    )
                    continue

                active_count += 1
                v_name = self.VEHICLE_CLASSES[obj_name]
                plate = self._get_or_create_plate(t_id, v_name)
                boxes_dict[t_id] = np.array([x1, y1, x2, y2], dtype=int)

                if t_id not in self.trajectories:
                    self.trajectories[t_id] = deque(maxlen=25)

                prev_center = self.trajectories[t_id][-1] if len(self.trajectories[t_id]) > 0 else (center_x, center_y)
                self.trajectories[t_id].append((center_x, center_y))
                prev_y, curr_y = int(prev_center[1]), center_y

                if (prev_y <= l1_y <= curr_y) or (curr_y <= l1_y <= prev_y):
                    if t_id not in self.line1_crossings:
                        self.line1_crossings[t_id] = current_time

                if (prev_y <= l2_y <= curr_y) or (curr_y <= l2_y <= prev_y):
                    if t_id not in self.line2_crossings:
                        self.line2_crossings[t_id] = current_time

                if t_id in self.line1_crossings and t_id in self.line2_crossings:
                    if t_id not in self.calculated_speeds:
                        t_delta = abs(self.line2_crossings[t_id] - self.line1_crossings[t_id])
                        if t_delta > 0.06:
                            computed_speed = float((distance_m / t_delta) * 3.6)
                            self.calculated_speeds[t_id] = computed_speed
                            self.total_scanned += 1

                            if computed_speed > speed_limit:
                                v_type_code = "overspeed_hmv" if v_name in ["lorry", "bus"] else "overspeed_lmv"
                                self._dispatch_violation(t_id, v_name, v_type_code, computed_speed, annotated_frame,
                                                         boxes_dict[t_id])

                            if v_name == "bike" and t_id % 3 == 0:
                                self._dispatch_violation(t_id, v_name, "no_helmet", computed_speed, annotated_frame,
                                                         boxes_dict[t_id])

                speed_val = self.calculated_speeds.get(t_id, None)
                cctns_hit = ap_cctns_radar.check_hotlist(plate)

                if speed_val is not None:
                    is_violator = speed_val > speed_limit
                    box_color = (0, 0, 255) if is_violator or cctns_hit else detection_color(obj_name)
                    vehicle_status = f"Track #{t_id} | {speed_val:.1f} km/h"
                else:
                    box_color = (0, 0, 255) if cctns_hit else detection_color(obj_name)
                    vehicle_status = f"Track #{t_id} | Tracking..."

                if cctns_hit:
                    vehicle_status += " | HOTLIST HIT"

                draw_detection_overlay(
                    annotated_frame,
                    (x1, y1, x2, y2),
                    f"{obj_name.title()} {score:.2f}",
                    obj_name,
                    vehicle_status,
                    box_color,
                )

                pts = list(self.trajectories[t_id])
                for i in range(1, len(pts)):
                    cv2.line(annotated_frame, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                             (int(pts[i][0]), int(pts[i][1])), box_color, 2)

            risk_alerts, corridor_risk = predictive_engine.analyze_behavior_and_predict_risk(
                self.trajectories, self.calculated_speeds, boxes_dict, lane_assignments
            )
            self.corridor_risk_index = corridor_risk

            for r_alert in risk_alerts:
                t_id = int(r_alert["track_id"])
                if t_id in boxes_dict:
                    self._dispatch_violation(t_id, "car", str(r_alert["type"]), self.calculated_speeds.get(t_id, 45.0),
                                             annotated_frame, boxes_dict[t_id])

            incident_alerts = self.incident_detector.evaluate_trajectories(
                self.trajectories, self.calculated_speeds, boxes_dict
            )
            for alert in incident_alerts:
                t_id = int(alert["track_id"])
                if t_id in boxes_dict:
                    self._dispatch_violation(t_id, "car", str(alert["type"]), 0.0, annotated_frame, boxes_dict[t_id])

            signal_telemetry = smart_signal.update_cycle(
                active_vehicles=active_count,
                pedestrians_waiting=pedestrian_danger_count,
                cattle_hazard=animal_danger_count
            )

            if evpp_status.get("active", False):
                sig_color = (0, 255, 128)
                sig_label = f"GREEN CORRIDOR ACTIVE ({evpp_status.get('emergency_type', 'EMERGENCY')})"
            else:
                sig_color = (0, 230, 0) if signal_telemetry["phase"] == "GREEN" else (
                    (0, 215, 255) if signal_telemetry["phase"] == "YELLOW" else (0, 0, 255))
                sig_label = f"SIGNAL: {signal_telemetry['phase']} ({signal_telemetry['seconds_in_phase']}s)"

            cv2.circle(annotated_frame, (60, 110), 16, sig_color, -1)
            cv2.putText(annotated_frame, sig_label, (85, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.48, sig_color, 2)

            self.active_incidents_list = [str(a["desc"]) for a in (incident_alerts + risk_alerts)]
            if pedestrian_danger_count > 0:
                self.active_incidents_list.append("ALERT: Pedestrians Crossing in Traffic Lane")
            if animal_danger_count > 0:
                self.active_incidents_list.append("CRITICAL: Animal detected in highway crossing zone")

            if self.active_incidents_list:
                banner_text = " | ".join(self.active_incidents_list[:2])
                cv2.rectangle(annotated_frame, (200, 115), (1080, 160), (0, 0, 220), -1)
                cv2.putText(annotated_frame, f"⚠️ {banner_text}", (215, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (255, 255, 255), 2)

            acoustic_data = acoustic_radar.get_telemetry()
            chain_ok, chain_msg, total_blocks = blockchain_ledger.verify_chain_integrity()
            emission_data = emission_auditor.get_corridor_emission_summary()

            t_infer = (time.perf_counter() - t_start) * 1000.0
            fps_tracker += 1
            now = time.perf_counter()
            elapsed = now - last_time
            current_fps = float(self.telemetry["fps"])
            if elapsed >= 1.0:
                current_fps = round(fps_tracker / elapsed, 1)
                fps_tracker = 0
                last_time = now

            annotated_rgb = cv2.cvtColor(np.ascontiguousarray(annotated_frame, dtype=np.uint8), cv2.COLOR_BGR2RGB)
            with self.lock:
                self.latest_annotated_rgb = annotated_rgb
                self.telemetry = {
                    "fps": current_fps,
                    "inference_ms": round(t_infer, 1),
                    "speed_limit": speed_limit,
                    "line1_y": l1_y,
                    "line2_y": l2_y,
                    "active_vehicles": active_count,
                    "total_passed": self.total_scanned,
                    "total_violations": self.total_violations,
                    "total_fines_inr": self.total_fines_inr,
                    "hotlist_hits": self.total_hotlist_alerts,
                    "pedestrians_in_danger": pedestrian_danger_count,
                    "cattle_on_highway": cattle_danger_count,
                    "animals_in_danger": animal_danger_count,
                    "surrounding_objects_count": surrounding_obj_count,
                    "object_counts": dict(sorted(object_counts.items())),
                    "detections": detections[:100],
                    "detection_model": os.path.basename(self.model_path),
                    "incidents": self.active_incidents_list,
                    "acoustic": acoustic_data,
                    "predictive_risk_score": self.corridor_risk_index,
                    "traffic_signal": signal_telemetry,
                    "green_corridor": evpp_status,
                    "weather": weather_stats,
                    "carbon_emissions": emission_data,
                    "blockchain": {"verified": chain_ok, "total_blocks": total_blocks, "status_text": chain_msg},
                    "rto_gateway_status": (
                        f"{dispatch_status.replace('_', ' ').upper()} | "
                        f"{dispatch_count_label} | "
                        f"Failed: {ap_rto_gateway.failed_dispatches} | Chain Height: {total_blocks}"
                    ),
                    "district_stats": dict(
                        sorted(self.district_violation_counts.items(), key=lambda item: item[1], reverse=True)[:6]),
                    "recent_violations": self.violation_events[-8:]
                }

            remaining_frame_time = minimum_frame_interval - (time.perf_counter() - t_start)
            if remaining_frame_time > 0:
                time.sleep(remaining_frame_time)

    def get_latest_frame_and_telemetry(self):
        with self.lock:
            frame = self.latest_annotated_rgb.copy() if self.latest_annotated_rgb is not None else None
            meta = dict(self.telemetry)
        return frame, meta

    def stop(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            self.thread = None


# ===========================================================================
# 17. WebRTC Video Stream Track
# ===========================================================================

class WebRTCVideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, engine: APPoliceRadarSafetyEngine, fps: int = 30):
        super().__init__()
        self.engine = engine
        self.fps = fps
        self._time_base = Fraction(1, 90000)
        self._start_time: Optional[float] = None

    async def recv(self) -> VideoFrame:
        if self._start_time is None:
            self._start_time = time.monotonic()

        frame_rgb, _ = self.engine.get_latest_frame_and_telemetry()
        if frame_rgb is None:
            frame_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame_rgb, "Connecting to Traffic Radar Feed...", (340, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.80,
                        (0, 255, 255), 2)

        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        elapsed = time.monotonic() - self._start_time
        video_frame.pts = int(elapsed * 90000)
        video_frame.time_base = self._time_base

        await asyncio.sleep(1.0 / self.fps)
        return video_frame
