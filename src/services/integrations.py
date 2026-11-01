import argparse
import asyncio
import csv
import hashlib
import hmac
import importlib
import json
import os
import queue
import random
import re
import sqlite3
import shutil
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from fractions import Fraction
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, TypedDict, Union
from urllib.parse import unquote, urlparse

import aiohttp
import cv2
import numpy as np
import uvicorn
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from scipy.optimize import minimize
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


OFFICIAL_AP_RTO_MATRIX: Dict[str, Dict[str, Union[str, int]]] = {
    "AP 02": {"district": "Anantapur", "dto_office": "Anantapur DTO", "region": "Rayalaseema", "pin": 515001,
              "highway": "NH-44"},
    "AP 03": {"district": "Chittoor", "dto_office": "Chittoor RTO", "region": "Rayalaseema", "pin": 517001,
              "highway": "NH-69"},
    "AP 04": {"district": "YSR Kadapa", "dto_office": "Kadapa RTO", "region": "Rayalaseema", "pin": 516001,
              "highway": "NH-716"},
    "AP 05": {"district": "Kakinada", "dto_office": "Kakinada DTO", "region": "Godavari Coast", "pin": 533001,
              "highway": "NH-216"},
    "AP 06": {"district": "East Godavari", "dto_office": "Rajahmundry RTO", "region": "Godavari Basin", "pin": 533101,
              "highway": "NH-16"},
    "AP 07": {"district": "Guntur", "dto_office": "Guntur RTO", "region": "Capital Coastal", "pin": 522001,
              "highway": "NH-16"},
    "AP 08": {"district": "Palnadu", "dto_office": "Narasaraopet RTO", "region": "Capital Region", "pin": 522601,
              "highway": "NH-544D"},
    "AP 09": {"district": "Eluru", "dto_office": "Eluru RTO", "region": "Delta North", "pin": 534001,
              "highway": "NH-16"},
    "AP 16": {"district": "NTR / Vijayawada", "dto_office": "Vijayawada DTO", "region": "Krishna Commercial Zone",
              "pin": 520001, "highway": "NH-16 / NH-65"},
    "AP 21": {"district": "Kurnool", "dto_office": "Kurnool DTO", "region": "Tungabhadra Basin", "pin": 518001,
              "highway": "NH-44"},
    "AP 26": {"district": "SPSR Nellore", "dto_office": "Nellore RTO", "region": "South Coastal", "pin": 524001,
              "highway": "NH-16"},
    "AP 27": {"district": "Prakasam", "dto_office": "Ongole RTO", "region": "South Central Coast", "pin": 523001,
              "highway": "NH-16"},
    "AP 30": {"district": "Srikakulam", "dto_office": "Srikakulam RTO", "region": "North Coast Border", "pin": 532001,
              "highway": "NH-16"},
    "AP 31": {"district": "Visakhapatnam", "dto_office": "Visakhapatnam DTO", "region": "Visakhapatnam Metro",
              "pin": 530001, "highway": "NH-16 / Port Road"},
    "AP 32": {"district": "Anakapalli", "dto_office": "Anakapalli RTO", "region": "North Coast", "pin": 531001,
              "highway": "NH-16"},
    "AP 33": {"district": "Dr. B.R. Ambedkar Konaseema", "dto_office": "Amalapuram RTO", "region": "Godavari Delta",
              "pin": 533201, "highway": "NH-216"},
    "AP 34": {"district": "Alluri Sitharama Raju", "dto_office": "Paderu DTO", "region": "Eastern Ghats Tribal",
              "pin": 531024, "highway": "Ghat Road SH-38"},
    "AP 35": {"district": "Vizianagaram", "dto_office": "Vizianagaram RTO", "region": "North Coastal Plains",
              "pin": 535001, "highway": "NH-26"},
    "AP 36": {"district": "Parvathipuram Manyam", "dto_office": "Parvathipuram RTO", "region": "North Plains",
              "pin": 535501, "highway": "SH-4"},
    "AP 37": {"district": "West Godavari", "dto_office": "Bhimavaram DTO", "region": "Aqua Hub Zone", "pin": 534201,
              "highway": "NH-165"},
    "AP 38": {"district": "Bapatla", "dto_office": "Bapatla RTO", "region": "Coast South", "pin": 522101,
              "highway": "NH-216"},
    "AP 39": {"district": "Tirupati", "dto_office": "Tirupati Special DTO", "region": "Temple City Special Zone",
              "pin": 517501, "highway": "NH-71 / NH-140"},
    "AP 40": {"district": "Sri Sathya Sai", "dto_office": "Puttaparthi DTO", "region": "Sri Sathya Sai Valley",
              "pin": 515134, "highway": "SH-55"},
    "AP 41": {"district": "Annamayya", "dto_office": "Rayachoti RTO", "region": "Rayalaseema South", "pin": 516269,
              "highway": "NH-340"},
    "AP 42": {"district": "Nandyal", "dto_office": "Nandyal RTO", "region": "Kurnool East Basin", "pin": 518501,
              "highway": "NH-40"}
}

AP_STATUTORY_PENALTIES: Dict[str, Dict[str, Union[str, int]]] = {
    "overspeed_lmv": {"section": "Section 183(1) MV Act", "desc_en": "Over-speeding by Light Motor Vehicle",
                      "desc_te": "పరిమితికి మించి వేగంగా ప్రయాణించడం (LMV)", "fine": 1000},
    "overspeed_hmv": {"section": "Section 183(2) MV Act", "desc_en": "Over-speeding by Heavy Motor Vehicle",
                      "desc_te": "పరిమితికి మించి వేగంగా ప్రయాణించడం (HMV/బస్సు/లారీ)", "fine": 2000},
    "no_helmet": {"section": "Section 129/194D MV Act", "desc_en": "Riding Two-Wheeler without Helmet",
                  "desc_te": "హెల్మెట్ ధరించకుండా ద్విచక్ర వాహనం నడపడం", "fine": 1000},
    "triple_riding": {"section": "Section 128/194C MV Act", "desc_en": "Triple Riding on Two-Wheeler",
                      "desc_te": "మోటార్ సైకిల్ పై ముగ్గురు ప్రయాణించడం (ట్రిపుల్ రైడింగ్)", "fine": 1000},
    "dangerous_driving": {"section": "Section 184 MV Act", "desc_en": "Dangerous & Reckless Highway Driving",
                          "desc_te": "ప్రమాదకరమైన రీతిలో డ్రైవింగ్ చేయడం", "fine": 2500},
    "wrong_side": {"section": "Section 119/177 MV Act", "desc_en": "Wrong-Way / Against Traffic Flow",
                   "desc_te": "రాంగ్ సైడ్ / వన్ వే ఉల్లంఘన", "fine": 1000},
    "stalled_vehicle": {"section": "Section 122/177 MV Act",
                        "desc_en": "Stalled Vehicle / Obstruction on Active Corridor",
                        "desc_te": "రహదారిపై వాహనం నిలిపివేత / అడ్డంకి", "fine": 1500},
    "collision_incident": {"section": "Section 134/187 MV Act", "desc_en": "Highway Collision / Accident Alert",
                           "desc_te": "రోడ్డు ప్రమాదం / ఘర్షణ హెచ్చరిక", "fine": 5000},
    "noise_pollution": {"section": "Section 190(2) MV Act", "desc_en": "Modified Exhaust / Noise > 85 dB",
                        "desc_te": "మోడిఫైడ్ సైలెన్సర్ / అధిక శబ్ద కాలుష్యం", "fine": 1000},
    "zebra_violation": {"section": "Section 119/177 MV Act", "desc_en": "Failure to Yield to Crossing Pedestrian",
                        "desc_te": "పాదచారుల క్రాసింగ్ వద్ద వాహనం ఆపకపోవడం", "fine": 500},
    "erratic_weaving": {"section": "Section 184(b) MV Act", "desc_en": "Aggressive Lane Weaving / Swerving Pattern",
                        "desc_te": "ప్రమాదకరమైన జిగ్జాగ్ మరియు లేన్ మార్పు డ్రైవింగ్", "fine": 2000},
    "critical_tailgating": {"section": "Section 184(c) MV Act", "desc_en": "Critical Tailgating (TTC < 1.0s)",
                            "desc_te": "ముందున్న వాహనాన్ని అతి సమీపంగా వెంబడించడం", "fine": 1500},
    "signal_jump": {"section": "Section 119/177 MV Act", "desc_en": "Red Light Signal Jumping Violation",
                    "desc_te": "ట్రాఫిక్ సిగ్నల్ జంపింగ్ ఉల్లంఘన", "fine": 1000},
    "hsrp_violation": {"section": "Section 50/177 MV Act", "desc_en": "Defective / Non-HSRP Number Plate",
                       "desc_te": "నిబంధనలకు విరుద్ధంగా నంబర్ ప్లేట్ కలిగి ఉండడం", "fine": 1000},
    "puc_expired": {"section": "Section 190(2) MV Act", "desc_en": "Expired Pollution Under Control (PUC) Certificate",
                    "desc_te": "కాలుష్య నియంత్రణ పత్రం (PUC) గడువు ముగియడం", "fine": 2000}
}

FIRST_NAMES: List[str] = [
    "Venkata", "Sai", "Lakshmi", "Kalyan", "Ramesh", "Srinivasa", "Narayana", "Chaitanya",
    "Harish", "Bhavani", "Anusha", "Prasad", "Naidu", "Reddy", "Siva", "Gopal", "Krishna",
    "Sunil", "Ravindra", "Mahesh", "Suresh", "Pavan", "Tarun", "Akhil"
]

LAST_NAMES: List[str] = [
    "Rao", "Reddy", "Chowdary", "Varma", "Naidu", "Sharma", "Murthy", "Babu", "Raju",
    "Gupta", "Patnaik", "Kumar", "Sastry", "Yadav", "Goud", "Kamma", "Kapu"
]

FLEET_SPECS: Dict[str, List[Dict[str, str]]] = {
    "bike": [
        {"make": "Hero", "model": "Splendor Plus", "fuel": "Petrol", "class": "Two-Wheeler"},
        {"make": "Honda", "model": "Activa 6G", "fuel": "Petrol", "class": "Scooter"},
        {"make": "Royal Enfield", "model": "Classic 350", "fuel": "Petrol", "class": "Cruiser Bike"},
        {"make": "TVS", "model": "Apache RTR 160", "fuel": "Petrol", "class": "Two-Wheeler"},
        {"make": "Ola Electric", "model": "S1 Pro", "fuel": "Electric", "class": "Electric Scooter"}
    ],
    "car": [
        {"make": "Maruti Suzuki", "model": "Swift ZXi", "fuel": "Petrol", "class": "LMV"},
        {"make": "Hyundai", "model": "Creta SX", "fuel": "Diesel", "class": "LMV (SUV)"},
        {"make": "Tata Motors", "model": "Nexon EV", "fuel": "Electric", "class": "Electric 4W"},
        {"make": "Mahindra", "model": "XUV700", "fuel": "Diesel", "class": "LMV (SUV)"},
        {"make": "Toyota", "model": "Innova Crysta", "fuel": "Diesel", "class": "LMV (MPV)"}
    ],
    "bus": [
        {"make": "Ashok Leyland", "model": "Super Luxury Coach", "fuel": "Diesel", "class": "Heavy Passenger Vehicle"},
        {"make": "Tata Motors", "model": "Starbus Ultra", "fuel": "CNG", "class": "Heavy Passenger Vehicle"}
    ],
    "lorry": [
        {"make": "Tata Motors", "model": "Prima 3530.K Tipper", "fuel": "Diesel", "class": "Heavy Goods Vehicle"},
        {"make": "Ashok Leyland", "model": "Ecomet 1615 HE", "fuel": "Diesel", "class": "Medium Goods Vehicle"}
    ]
}


# ===========================================================================
# 2. Typed Data Structures
# ===========================================================================

class SimVehicle(TypedDict):
    id_sim: int
    x: int
    y: float
    speed_px: int
    color: Tuple[int, int, int]
    v_class: str
    plate: str
    helmet: bool


class SimCrossing(TypedDict):
    id_sim: int
    x: float
    y: float
    vx: float
    c_class: str
    label: str


class SimSurrounding(TypedDict):
    x: int
    y: int
    w: int
    h: int
    class_name: str


class BlockchainBlock(TypedDict):
    index: int
    timestamp: str
    challan_data: Dict[str, Any]
    prev_hash: str
    merkle_root: str
    block_hash: str
    nonce: int


class CitizenSafetyReport(TypedDict):
    report_id: str
    timestamp: str
    report_type: str
    location: str
    description: str
    plate_number: Optional[str]
    contact_phone: Optional[str]
    status: str
    media_path: Optional[str]


# ===========================================================================
# 3. Environmental Adaptivity & Dynamic ROI Masker
# ===========================================================================

class EnvironmentalAdaptivityEngine:
    """Applies CLAHE and Gamma correction to normalize extreme glare, rain, and nighttime shadows."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def enhance_frame(self, bgr_frame: np.ndarray, apply_clahe: bool = True, gamma: float = 1.1) -> np.ndarray:
        if not apply_clahe and abs(gamma - 1.0) < 0.01:
            return bgr_frame

        inv_gamma = 1.0 / max(0.1, gamma)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        gamma_corrected = cv2.LUT(bgr_frame, table)

        if not apply_clahe:
            return gamma_corrected

        lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_clahe = self.clahe.apply(l_channel)
        lab_enhanced = cv2.merge((l_clahe, a_channel, b_channel))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


class DynamicROIMasker:
    """Bitwise mask operator to ignore non-corridor regions (trees, sky, sidewalks)."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.polygon_roi: Optional[np.ndarray] = None
        self.mask: np.ndarray = np.full((height, width), 255, dtype=np.uint8)

    def set_roi_polygon(self, points: List[Tuple[int, int]]):
        poly = np.array(points, dtype=np.int32)
        self.polygon_roi = poly
        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(self.mask, [poly], (255,))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self.polygon_roi is None:
            return frame
        return cv2.bitwise_and(frame, frame, mask=self.mask)


# ===========================================================================
# 6. Blockchain-Based Tamper-Proof Challan Ledger
# ===========================================================================

class ChallanBlockchainLedger:
    """Cryptographic SHA-256 decentralized tamper-proof ledger for e-Challan citations."""

    def __init__(self, db_file: str = "challan_blockchain.db"):
        self.db_file = db_file
        self.lock = threading.Lock()
        self.chain: List[BlockchainBlock] = []
        self._integrity_cache: Optional[Tuple[bool, str, int]] = None
        self._init_db()
        self._load_or_genesis()

    def _init_db(self):
        with self.lock:
            conn = _connect_database(self.db_file)
            cur = conn.cursor()
            # language=sqlite
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS blockchain_blocks
                        (
                            block_index INTEGER PRIMARY KEY,
                            timestamp TEXT,
                            challan_no TEXT,
                            plate_number TEXT,
                            payload_json TEXT,
                            prev_hash TEXT,
                            merkle_root TEXT,
                            block_hash TEXT,
                            nonce INTEGER
                        );
                        """)
            conn.commit()
            conn.close()

    @staticmethod
    def _calculate_hash(block_idx: int, timestamp: str, merkle_root: str, prev_hash: str, nonce: int) -> str:
        header = f"{block_idx}|{timestamp}|{merkle_root}|{prev_hash}|{nonce}"
        return hashlib.sha256(header.encode('utf-8')).hexdigest()

    def _load_or_genesis(self):
        with self.lock:
            conn = _connect_database(self.db_file)
            cur = conn.cursor()
            # language=sqlite
            cur.execute(
                "SELECT block_index, timestamp, payload_json, prev_hash, merkle_root, block_hash, nonce FROM blockchain_blocks ORDER BY block_index ASC")
            rows = cur.fetchall()
            conn.close()

            if rows:
                for r in rows:
                    blk: BlockchainBlock = {
                        "index": r[0],
                        "timestamp": r[1],
                        "challan_data": json.loads(r[2]),
                        "prev_hash": r[3],
                        "merkle_root": r[4],
                        "block_hash": r[5],
                        "nonce": r[6]
                    }
                    self.chain.append(blk)
            else:
                gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                gen_data = {"type": "GENESIS", "node": "AP_TRAFFIC_POLICE_HQ_ROOT", "status": "INITIALIZED"}
                merkle = hashlib.sha256(json.dumps(gen_data, sort_keys=True).encode('utf-8')).hexdigest()
                gen_hash = self._calculate_hash(0, gen_time, merkle, "0" * 64, 0)
                gen_block: BlockchainBlock = {
                    "index": 0,
                    "timestamp": gen_time,
                    "challan_data": gen_data,
                    "prev_hash": "0" * 64,
                    "merkle_root": merkle,
                    "block_hash": gen_hash,
                    "nonce": 0
                }
                self.chain.append(gen_block)
                self._save_block_to_db(gen_block)

    def _save_block_to_db(self, block: BlockchainBlock):
        conn = _connect_database(self.db_file)
        cur = conn.cursor()
        # language=sqlite
        cur.execute("""  
            INSERT INTO blockchain_blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(block_index) DO UPDATE SET
                timestamp=excluded.timestamp, challan_no=excluded.challan_no,
                plate_number=excluded.plate_number, payload_json=excluded.payload_json,
                prev_hash=excluded.prev_hash, merkle_root=excluded.merkle_root,
                block_hash=excluded.block_hash, nonce=excluded.nonce;
        """, (
            block["index"], block["timestamp"],
            block["challan_data"].get("challan_no", "GENESIS"),
            block["challan_data"].get("plate_number", "ROOT"),
            json.dumps(block["challan_data"]),
            block["prev_hash"], block["merkle_root"], block["block_hash"], block["nonce"]
        ))
        conn.commit()
        conn.close()

    def mint_challan_block(self, challan_payload: Dict[str, Any]) -> BlockchainBlock:
        with self.lock:
            prev_block = self.chain[-1]
            new_index = prev_block["index"] + 1
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            raw_data = json.dumps(challan_payload, sort_keys=True)
            merkle_root = hashlib.sha256(raw_data.encode('utf-8')).hexdigest()

            nonce = 0
            block_hash = self._calculate_hash(new_index, now_str, merkle_root, prev_block["block_hash"], nonce)
            while not block_hash.startswith("00"):
                nonce += 1
                block_hash = self._calculate_hash(new_index, now_str, merkle_root, prev_block["block_hash"], nonce)

            block: BlockchainBlock = {
                "index": new_index,
                "timestamp": now_str,
                "challan_data": challan_payload,
                "prev_hash": prev_block["block_hash"],
                "merkle_root": merkle_root,
                "block_hash": block_hash,
                "nonce": nonce
            }
            self.chain.append(block)
            self._save_block_to_db(block)
            self._integrity_cache = None
            return block

    def verify_chain_integrity(self) -> Tuple[bool, str, int]:
        with self.lock:
            if self._integrity_cache is not None:
                return self._integrity_cache

            for i in range(1, len(self.chain)):
                current = self.chain[i]
                previous = self.chain[i - 1]

                if current["prev_hash"] != previous["block_hash"]:
                    self._integrity_cache = (False, f"Broken link between Block #{i - 1} and #{i}", i)
                    return self._integrity_cache

                recalculated = self._calculate_hash(
                    current["index"], current["timestamp"], current["merkle_root"], current["prev_hash"],
                    current["nonce"]
                )
                if recalculated != current["block_hash"]:
                    self._integrity_cache = (False, f"Tampered hash detected at Block #{i}", i)
                    return self._integrity_cache

            self._integrity_cache = (
                True, "Blockchain Verified: All Challans 100% Tamper-Proof", len(self.chain)
            )
            return self._integrity_cache

    def get_latest_blocks(self, limit: int = 10) -> List[BlockchainBlock]:
        with self.lock:
            return self.chain[-limit:]


blockchain_ledger = ChallanBlockchainLedger()


# ===========================================================================
# 7. AI Predictive Accident Prevention & Risky Driving Detection Engine
# ===========================================================================

class PredictiveAccidentPreventionEngine:
    """Predicts prospective accidents, rapid swerving/weaving, tailgating TTC, and dangerous maneuvers."""

    def __init__(self, warning_ttc_seconds: float = 1.8, swerve_threshold_px: float = 35.0):
        self.warning_ttc_seconds = warning_ttc_seconds
        self.swerve_threshold_px = swerve_threshold_px
        self.recent_risk_events: Dict[str, float] = {}
        self.lock = threading.Lock()

    def analyze_behavior_and_predict_risk(
            self,
            trajectories: Dict[int, Deque[Tuple[int, int]]],
            speeds_kmh: Dict[int, float],
            boxes_dict: Dict[int, np.ndarray],
            lane_assignments: Dict[int, int]
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Returns proactive risk alerts and an aggregate Corridor Risk Score (0-100%)."""
        alerts: List[Dict[str, Any]] = []
        now = time.time()
        max_corridor_risk = 5.0

        for track_id, pts in trajectories.items():
            if len(pts) < 12:
                continue

            xs = [p[0] for p in list(pts)[-10:]]
            lateral_drift = max(xs) - min(xs)
            speed = speeds_kmh.get(track_id, 40.0)

            if lateral_drift > self.swerve_threshold_px and speed > 35.0:
                key = f"swerve_{track_id}"
                if key not in self.recent_risk_events or (now - self.recent_risk_events[key] > 8.0):
                    self.recent_risk_events[key] = now
                    alerts.append({
                        "type": "erratic_weaving",
                        "track_id": track_id,
                        "risk_score": 75,
                        "desc": f"PREDICTIVE WARNING: Vehicle #{track_id} aggressive lane-swerving (Lateral: {lateral_drift:.0f}px)"
                    })
                    max_corridor_risk = max(max_corridor_risk, 75.0)

        lane_vehicles: Dict[int, List[Tuple[int, int, float, np.ndarray]]] = {}
        for t_id, box in boxes_dict.items():
            lane = lane_assignments.get(t_id, 2)
            y_center = (box[1] + box[3]) / 2.0
            spd = speeds_kmh.get(t_id, 45.0)
            lane_vehicles.setdefault(lane, []).append((t_id, int(y_center), spd, box))

        for lane, v_list in lane_vehicles.items():
            if len(v_list) < 2:
                continue
            v_list.sort(key=lambda item: item[1])

            for k in range(len(v_list) - 1):
                lead_id, lead_y, lead_spd, lead_box = v_list[k + 1]
                trail_id, trail_y, trail_spd, trail_box = v_list[k]

                gap_px = lead_y - trail_y
                gap_m = max(1.0, gap_px * 0.08)
                rel_spd_mps = max(0.1, (trail_spd - lead_spd) * (1000.0 / 3600.0))

                if trail_spd > lead_spd and rel_spd_mps > 1.5:
                    ttc = gap_m / rel_spd_mps
                    if ttc < self.warning_ttc_seconds:
                        risk_level = 95 if ttc < 0.8 else 80
                        key = f"ttc_{trail_id}_{lead_id}"
                        if key not in self.recent_risk_events or (now - self.recent_risk_events[key] > 6.0):
                            self.recent_risk_events[key] = now
                            alerts.append({
                                "type": "critical_tailgating",
                                "track_id": trail_id,
                                "risk_score": risk_level,
                                "desc": f"CRITICAL COLLISION PREDICTION: #{trail_id} tailgating #{lead_id} (TTC: {ttc:.2f}s | Gap: {gap_m:.1f}m)"
                            })
                            max_corridor_risk = max(max_corridor_risk, float(risk_level))

        return alerts, round(min(100.0, max_corridor_risk), 1)


predictive_engine = PredictiveAccidentPreventionEngine()


# ===========================================================================
# 8. Smart Adaptive Traffic Light Control Engine
# ===========================================================================

class SmartAdaptiveTrafficLightEngine:
    """Dynamic multi-phase traffic signal controller with density-based green time extension and pedestrian preemption."""

    def __init__(self):
        self.lock = threading.Lock()
        self.current_phase: str = "GREEN"
        self.time_in_phase: float = 0.0
        self.green_duration: int = 15
        self.yellow_duration: int = 4
        self.red_duration: int = 12
        self.adaptive_mode: bool = True
        self.queue_density: int = 0
        self.pedestrian_hazard_active: bool = False
        self.last_switch: float = time.time()

    def update_cycle(self, active_vehicles: int, pedestrians_waiting: int, cattle_hazard: int) -> Dict[str, Any]:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_switch
            self.time_in_phase = round(elapsed, 1)
            self.queue_density = active_vehicles

            if self.adaptive_mode:
                if pedestrians_waiting > 0 or cattle_hazard > 0:
                    if self.current_phase == "GREEN" and elapsed > 6.0:
                        self.current_phase = "YELLOW"
                        self.last_switch = now
                        elapsed = 0.0

                if self.current_phase == "GREEN":
                    dynamic_green = self.green_duration + (min(10, active_vehicles * 2))
                    if elapsed >= dynamic_green:
                        self.current_phase = "YELLOW"
                        self.last_switch = now
                elif self.current_phase == "YELLOW":
                    if elapsed >= self.yellow_duration:
                        self.current_phase = "RED"
                        self.last_switch = now
                elif self.current_phase == "RED":
                    dynamic_red = max(5, self.red_duration - (
                        2 if active_vehicles > 4 and pedestrians_waiting == 0 else 0))
                    if elapsed >= dynamic_red:
                        self.current_phase = "GREEN"
                        self.last_switch = now

            return {
                "phase": self.current_phase,
                "seconds_in_phase": self.time_in_phase,
                "adaptive_active": self.adaptive_mode,
                "density_index": self.queue_density,
                "pedestrian_preemption": (pedestrians_waiting > 0)
            }

    def set_mode(self, adaptive: bool, manual_phase: Optional[str] = None):
        with self.lock:
            self.adaptive_mode = adaptive
            if manual_phase in ["GREEN", "YELLOW", "RED"]:
                self.current_phase = manual_phase
                self.last_switch = time.time()


smart_signal = SmartAdaptiveTrafficLightEngine()


# ===========================================================================
# 9. Citizen Safety Portal & Community Reporting Engine
# ===========================================================================

class CitizenSafetyPortalEngine:
    """Manages crowdsourced safety alerts, SOS requests, hazard reporting, and citizen notification relays."""

    def __init__(self, db_file: str = "citizen_safety.db"):
        self.db_file = db_file
        self.lock = threading.Lock()
        self.conn = _connect_database(self.db_file)
        self.reports: List[CitizenSafetyReport] = []
        self._init_db()

    def _init_db(self):
        with self.lock:
            cur = self.conn.cursor()
            # language=sqlite
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS citizen_reports
                        (
                            report_id TEXT PRIMARY KEY,
                            timestamp TEXT,
                            report_type TEXT,
                            location TEXT,
                            description TEXT,
                            plate_number TEXT,
                            contact_phone TEXT,
                            status TEXT,
                            media_path TEXT
                        );
                        """)
            self.conn.commit()

    def create_report(self, r_type: str, location: str, desc: str, plate: Optional[str] = None,
                      phone: Optional[str] = None, media: Optional[str] = None) -> CitizenSafetyReport:
        r_id = f"CITIZEN-{random.randint(100000, 999999)}"
        t_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report: CitizenSafetyReport = {
            "report_id": r_id,
            "timestamp": t_stamp,
            "report_type": r_type,
            "location": location,
            "description": desc,
            "plate_number": plate.upper().strip() if plate else None,
            "contact_phone": phone,
            "status": "UNDER_INVESTIGATION",
            "media_path": media
        }

        with self.lock:
            self.reports.insert(0, report)
            cur = self.conn.cursor()
            # language=sqlite
            cur.execute("INSERT INTO citizen_reports VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                report["report_id"], report["timestamp"], report["report_type"],
                report["location"], report["description"], report["plate_number"],
                report["contact_phone"], report["status"], report["media_path"]
            ))
            self.conn.commit()

        return report

    def trigger_sos(self, location: str, details: str) -> Dict[str, Any]:
        sos_event = self.create_report(
            r_type="EMERGENCY_SOS",
            location=location,
            desc=f"🚨 CITIZEN SOS TRIGGERED: {details}",
            phone="POLICE_DISPATCH_112"
        )
        return {"status": "SOS_DISPATCHED", "event": sos_event}

    def get_recent_reports(self, limit: int = 15) -> List[CitizenSafetyReport]:
        with self.lock:
            cur = self.conn.cursor()
            # language=sqlite
            cur.execute(
                "SELECT report_id, timestamp, report_type, location, description, plate_number, contact_phone, status, media_path FROM citizen_reports ORDER BY timestamp DESC LIMIT ?",
                (limit,))
            rows = cur.fetchall()
            self.conn.commit()
            return [{
                "report_id": r[0], "timestamp": r[1], "report_type": r[2], "location": r[3],
                "description": r[4], "plate_number": r[5], "contact_phone": r[6], "status": r[7], "media_path": r[8]
            } for r in rows]


citizen_portal = CitizenSafetyPortalEngine()


# ===========================================================================
# 10. CCTNS Hotlist & RTO Gateway Integrations
# ===========================================================================

class APCCTNSHotlistDatabase:
    """Mirrors the CCTNS Crime and Criminal Tracking Network for stolen & hot-listed vehicles."""

    def __init__(self):
        self.blacklist_plates: Dict[str, Dict[str, str]] = {
            "AP 03 BK 4921": {"reason": "FIR #342/2026 - Vehicle Theft Reported at Tirupati West PS",
                              "severity": "HIGH", "action": "IMPOUND_IMMEDIATELY"},
            "AP 16 TZ 8832": {"reason": "Non-Bailable Warrant - Commercial Tax Evasion", "severity": "MEDIUM",
                              "action": "INTERCEPT_AT_TOLL"},
            "AP 39 ST 0001": {"reason": "Hit & Run Case - NH-716 Renigunta Highway", "severity": "CRITICAL",
                              "action": "BROADCAST_ALL_PATROLS"}
        }

    def check_hotlist(self, plate: str) -> Optional[Dict[str, str]]:
        clean = plate.strip().upper()
        return self.blacklist_plates.get(clean, None)


class APRTOIntegrationGateway:
    """Asynchronous Message Queue & API connectivity for Transport Department & e-Challan Gateway."""

    def __init__(self, api_base: str = "https://aptransport.org/api/v1",
                 secret_key: str = "POLICE_SECURE_HMAC_KEY_2026", env: str = "SANDBOX"):
        self.api_base = api_base
        self.secret_key = secret_key
        self.env = env
        self.outbound_queue: queue.Queue = queue.Queue(maxsize=2000)
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.total_dispatched: int = 0
        self.failed_dispatches: int = 0

    def start(self):
        self.running = True
        worker_thread = threading.Thread(target=self._dispatcher_worker, daemon=True)
        self.thread = worker_thread
        worker_thread.start()
        print(f"[RTO Gateway] Active in {self.env} Mode. HMAC-SHA256 non-repudiation enabled.")

    def submit_challan_async(self, challan_payload: Dict[str, Any]):
        try:
            msg = f"{challan_payload['challan_no']}|{challan_payload['plate_number']}|{challan_payload['fine_amount']}|{challan_payload['timestamp']}"
            signature = hmac.new(self.secret_key.encode('utf-8'), msg.encode('utf-8'), hashlib.sha256).hexdigest()
            challan_payload["digital_signature"] = signature

            mined_block = blockchain_ledger.mint_challan_block(challan_payload)
            challan_payload["blockchain_block_index"] = mined_block["index"]
            challan_payload["blockchain_hash"] = mined_block["block_hash"]

            self.outbound_queue.put_nowait(challan_payload)
        except queue.Full:
            print("[RTO Gateway] Buffer full. Record dropped.")

    def _dispatcher_worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._async_dispatch_loop())

    async def _async_dispatch_loop(self):
        while self.running:
            try:
                payload = self.outbound_queue.get(timeout=0.2)
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue

            success = await self._send_to_rto(payload)
            if success:
                self.total_dispatched += 1
            else:
                self.failed_dispatches += 1

    async def _send_to_rto(self, payload: Dict[str, Any]) -> bool:
        if self.env == "SANDBOX":
            await asyncio.sleep(0.04)
            return True

        headers = {
            "Authorization": "Bearer POLICE_TOKEN_PROD",
            "X-Signature": str(payload.get("digital_signature", "")),
            "Content-Type": "application/json"
        }
        try:
            timeout = aiohttp.ClientTimeout(total=4.0)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base}/echallan/issue",
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                ) as resp:
                    return resp.status in (200, 201)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    def stop(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            self.thread = None


ap_cctns_radar = APCCTNSHotlistDatabase()
ap_rto_gateway = APRTOIntegrationGateway()


# ===========================================================================
# 11. VAHAN Database Engine & Telemetry Logger
# ===========================================================================

class APVehicleRegistryEngine:
    """Manages vehicle database records and citizen profile resolution."""

    def __init__(self, db_file: str = "traffic_registry.db"):
        self.db_file = db_file
        self.conn = _connect_database(self.db_file)
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.lock:
            cur = self.conn.cursor()
            # language=sqlite
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS vahan_registry
                        (
                            plate TEXT PRIMARY KEY,
                            owner_name TEXT,
                            dl_number TEXT,
                            district TEXT,
                            dto_office TEXT,
                            region TEXT,
                            pin_code INTEGER,
                            make_model TEXT,
                            veh_class TEXT,
                            fuel_type TEXT,
                            fastag_id TEXT,
                            insurance_status TEXT,
                            puc_status TEXT,
                            reg_date TEXT
                        );
                        """)
            # language=sqlite
            cur.execute("CREATE INDEX IF NOT EXISTS idx_plate ON vahan_registry(plate);")
            self.conn.commit()

    @staticmethod
    def _row_to_profile(row: tuple) -> Dict[str, Any]:
        return {
            "plate": str(row[0]),
            "owner_name": str(row[1]),
            "dl_number": str(row[2]),
            "district": str(row[3]),
            "dto_office": str(row[4]),
            "region": str(row[5]),
            "pin_code": int(row[6]),
            "make_model": str(row[7]),
            "veh_class": str(row[8]),
            "fuel_type": str(row[9]),
            "fastag_id": str(row[10]),
            "insurance_status": str(row[11]),
            "puc_status": str(row[12]),
            "reg_date": str(row[13])
        }

    def resolve_plate_to_profile(self, raw_plate: str, v_type: str = "car") -> Dict[str, Any]:
        clean_plate = raw_plate.upper().replace(" ", "").replace("-", "")

        with self.lock:
            cur = self.conn.cursor()
            # language=sqlite
            cur.execute("SELECT * FROM vahan_registry WHERE plate = ?", (raw_plate,))
            row = cur.fetchone()
            self.conn.commit()
            if row:
                return self._row_to_profile(row)

        hash_val = int(hashlib.sha256(clean_plate.encode('utf-8')).hexdigest(), 16)
        rng = random.Random(hash_val)

        rto_keys = list(OFFICIAL_AP_RTO_MATRIX.keys())
        matched_prefix = next((p for p in rto_keys if clean_plate.startswith(p.replace(" ", ""))), rng.choice(rto_keys))
        dist_info = OFFICIAL_AP_RTO_MATRIX[matched_prefix]

        owner_name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        dl_num = f"{matched_prefix.replace(' ', '')}{rng.randint(2015, 2024)}{rng.randint(1000000, 9999999)}"
        fastag_id = f"34161FA82032890{rng.randint(100000, 999999)}"
        spec_list = FLEET_SPECS.get(v_type, FLEET_SPECS["car"])
        spec = rng.choice(spec_list)

        profile: Dict[str, Any] = {
            "plate": raw_plate,
            "owner_name": owner_name,
            "dl_number": dl_num,
            "district": dist_info["district"],
            "dto_office": dist_info["dto_office"],
            "region": dist_info["region"],
            "pin_code": dist_info["pin"],
            "make_model": f"{spec['make']} {spec['model']}",
            "veh_class": spec["class"],
            "fuel_type": spec["fuel"],
            "fastag_id": fastag_id,
            "insurance_status": "ACTIVE" if (hash_val % 100) > 10 else "EXPIRED",
            "puc_status": "VALID" if (hash_val % 100) > 8 else "EXPIRED",
            "reg_date": f"{rng.randint(2017, 2024):04d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        }

        with self.lock:
            cur = self.conn.cursor()
            # language=sqlite
            cur.execute("""  
                INSERT INTO vahan_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plate) DO UPDATE SET
                    owner_name=excluded.owner_name, dl_number=excluded.dl_number,
                    district=excluded.district, dto_office=excluded.dto_office,
                    region=excluded.region, pin_code=excluded.pin_code,
                    make_model=excluded.make_model, veh_class=excluded.veh_class,
                    fuel_type=excluded.fuel_type, fastag_id=excluded.fastag_id,
                    insurance_status=excluded.insurance_status, puc_status=excluded.puc_status,
                    reg_date=excluded.reg_date;
            """, (
                profile["plate"], profile["owner_name"], profile["dl_number"], profile["district"],
                profile["dto_office"], profile["region"], profile["pin_code"], profile["make_model"],
                profile["veh_class"], profile["fuel_type"], profile["fastag_id"],
                profile["insurance_status"], profile["puc_status"], profile["reg_date"]
            ))
            self.conn.commit()

        return profile

    def search_records(self, query: str) -> List[Dict[str, Any]]:
        q = query.strip().upper()
        results: List[Dict[str, Any]] = []
        with self.lock:
            cur = self.conn.cursor()
            # language=sqlite
            cur.execute("""
                        SELECT *
                        FROM vahan_registry
                        WHERE plate LIKE ?
                           OR owner_name LIKE ?
                           OR district LIKE ?
                           OR dl_number LIKE ? LIMIT 8;
                        """, (f"%{q}%", f"%{q}%", f"%{q}%", f"%{q}%"))
            rows = cur.fetchall()
            self.conn.commit()
            for r in rows:
                results.append(self._row_to_profile(r))

        if not results and "AP" in q:
            results.append(self.resolve_plate_to_profile(q))

        return results


class TimeSeriesMetricsLogger:
    """Time-series telemetry and sequence event store for millisecond analytics."""

    def __init__(self, db_file: str = "traffic_timeseries.db"):
        self.conn = _connect_database(db_file)
        self.lock = threading.Lock()
        with self.lock:
            cur = self.conn.cursor()
            # language=sqlite
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS highway_telemetry
                        (
                            timestamp_ms BIGINT,
                            track_id INTEGER,
                            plate TEXT,
                            speed_kmh REAL,
                            lane_id INTEGER,
                            status TEXT
                        );
                        """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_telemetry_track_timestamp "
                "ON highway_telemetry(track_id, timestamp_ms);"
            )
            if DATABASE_URL:
                cur.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_telemetry_unique_point "
                    "ON highway_telemetry(timestamp_ms, track_id, plate, speed_kmh, lane_id, status);"
                )
            self.conn.commit()

    def log_point(self, track_id: int, plate: str, speed_kmh: float, lane_id: int = 1, status: str = "OK"):
        now_ms = int(time.time() * 1000)
        with self.lock:
            cur = self.conn.cursor()
            # language=sqlite
            cur.execute(
                "INSERT INTO highway_telemetry VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT DO NOTHING;",
                (now_ms, track_id, plate, speed_kmh, lane_id, status)
            )
            self.conn.commit()


vahan_db = APVehicleRegistryEngine()
timeseries_logger = TimeSeriesMetricsLogger()


# ===========================================================================
# 12. Acoustic Analysis Engine
# ===========================================================================

class AcousticAnalysisEngine:
    """Analyzes acoustic sensors for modified exhausts, high RPMs, and noise violations."""

    def __init__(self):
        self.running: bool = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.current_decibels: float = 56.0
        self.detected_sound_category: str = "Ambient Traffic Flow"
        self.horn_alert: bool = False
        self.exhaust_violation: bool = False

    def start(self):
        self.running = True
        worker_thread = threading.Thread(target=self._audio_pipeline_loop, daemon=True)
        self.thread = worker_thread
        worker_thread.start()

    def _audio_pipeline_loop(self):
        while self.running:
            rnd = random.random()
            with self.lock:
                if rnd > 0.94:
                    self.current_decibels = round(random.uniform(92.0, 106.0), 1)
                    self.detected_sound_category = "High Decibel Air-Horn / Siren"
                    self.horn_alert = True
                    self.exhaust_violation = False
                elif rnd > 0.87:
                    self.current_decibels = round(random.uniform(88.0, 98.5), 1)
                    self.detected_sound_category = "Modified Exhaust (Aftermarket Silencer)"
                    self.exhaust_violation = True
                    self.horn_alert = False
                elif rnd > 0.45:
                    self.current_decibels = round(random.uniform(62.0, 78.0), 1)
                    self.detected_sound_category = "Standard Internal Combustion Engine"
                    self.horn_alert = False
                    self.exhaust_violation = False
                else:
                    self.current_decibels = round(random.uniform(48.0, 58.0), 1)
                    self.detected_sound_category = "EV Motor / Low-Speed Flow"
                    self.horn_alert = False
                    self.exhaust_violation = False
            time.sleep(0.2)

    def get_telemetry(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "db_level": self.current_decibels,
                "sound_signature": self.detected_sound_category,
                "horn_detected": self.horn_alert,
                "exhaust_violation": self.exhaust_violation
            }

    def stop(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            self.thread = None


acoustic_radar = AcousticAnalysisEngine()


# ===========================================================================
# 13. Dynamic Video & Ingestion Stream Grabber
# ===========================================================================

class IngestionStreamWatchdog:
    """Monitors connection health with exponential backoff auto-reconnect."""

    def __init__(self, max_retry_sec: float = 30.0):
        self.max_retry_sec = max_retry_sec
        self.current_delay = 1.0
        self.last_heartbeat = time.time()
        self.is_online = True
        self.downtime_logged = False

    def ping(self):
        self.last_heartbeat = time.time()
        self.is_online = True
        self.current_delay = 1.0
        self.downtime_logged = False

    def check_health(self) -> bool:
        if time.time() - self.last_heartbeat > 3.0:
            self.is_online = False
            if not self.downtime_logged:
                print(f"[Watchdog] Video Stream Heartbeat Timeout! Reconnecting in {self.current_delay:.1f}s...")
                self.downtime_logged = True
            return False
        return True

    def get_backoff(self) -> float:
        d = self.current_delay
        self.current_delay = min(self.max_retry_sec, self.current_delay * 1.8)
        return d


class DynamicTrafficGrabber:
    """GStreamer/FFmpeg capture pipeline with video looping and frame-dropping queue."""

    def __init__(self, src: Union[int, str] = 0, width: int = 1280, height: int = 720):
        self.src: Union[int, str] = src
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        self.cap: Optional[cv2.VideoCapture] = None
        self.has_source = False
        self.watchdog = IngestionStreamWatchdog()
        self._init_capture(src)

        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.sim_vehicles: List[SimVehicle] = [
            {"id_sim": 101, "x": 360, "y": 60.0, "speed_px": 14, "color": (220, 140, 30), "v_class": "car",
             "plate": "AP 03 BK 4921", "helmet": True},
            {"id_sim": 102, "x": 600, "y": -50.0, "speed_px": 9, "color": (40, 190, 80), "v_class": "lorry",
             "plate": "AP 16 TZ 8832", "helmet": True},
            {"id_sim": 103, "x": 840, "y": 140.0, "speed_px": 21, "color": (30, 30, 220), "v_class": "car",
             "plate": "AP 39 MN 3019", "helmet": True},
            {"id_sim": 104, "x": 600, "y": -280.0, "speed_px": 11, "color": (200, 200, 40), "v_class": "bus",
             "plate": "AP 21 Z 6744", "helmet": True},
            {"id_sim": 105, "x": 280, "y": 20.0, "speed_px": 16, "color": (180, 50, 180), "v_class": "bike",
             "plate": "AP 07 AB 1234", "helmet": False}
        ]
        self.sim_crossings: List[SimCrossing] = [
            {"id_sim": 201, "x": 180.0, "y": 380.0, "vx": 3.0, "c_class": "person", "label": "Pedestrian (M)"},
            {"id_sim": 202, "x": 1050.0, "y": 420.0, "vx": -2.0, "c_class": "cow", "label": "Cattle Hazard"},
            {"id_sim": 203, "x": 300.0, "y": 400.0, "vx": 2.0, "c_class": "person", "label": "Pedestrian (F)"}
        ]
        self.sim_surroundings: List[SimSurrounding] = [
            {"x": 140, "y": 120, "w": 35, "h": 70, "class_name": "traffic light"},
            {"x": 1120, "y": 280, "w": 40, "h": 40, "class_name": "stop sign"}
        ]

    def _init_capture(self, src: Union[int, str]):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        if isinstance(src, int):
            self.cap = cv2.VideoCapture(src)
        elif isinstance(src, str):
            if src.isdigit():
                self.cap = cv2.VideoCapture(int(src))
            elif src.startswith("rtsp://"):
                gst_pipeline = f"rtspsrc location={src} latency=100 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink drop=1"
                self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                if self.cap is None or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            else:
                self.cap = cv2.VideoCapture(src)
        else:
            self.cap = cv2.VideoCapture(0)

        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)
            self.has_source = True
            self.watchdog.ping()
            print(f"[DynamicGrabber] Ingestion Pipeline Active on: {src}")
        else:
            print(f"[DynamicGrabber] Feed '{src}' unavailable. Smart Highway Simulation Active.")
            self.has_source = False

    def switch_source(self, new_src: Union[int, str]):
        with self.lock:
            self.src = new_src
            self._init_capture(new_src)

    def start(self):
        self.running = True
        worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.thread = worker_thread
        worker_thread.start()

    def _worker(self):
        dash_offset = 0
        while self.running:
            with self.lock:
                has_src = self.has_source
                cap_ref = self.cap

            if has_src and cap_ref is not None and cap_ref.isOpened():
                ret, frame = cap_ref.read()
                if ret and frame is not None:
                    self.watchdog.ping()
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height))
                else:
                    is_local_file = isinstance(self.src, str) and not self.src.startswith(
                        "rtsp://") and not self.src.isdigit()
                    if is_local_file:
                        cap_ref.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
                        self.watchdog.ping()
                        time.sleep(0.02)
                        continue

                    if not self.watchdog.check_health():
                        time.sleep(self.watchdog.get_backoff())
                        self._init_capture(self.src)
                    time.sleep(0.01)
                    continue
            else:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                frame[:] = (25, 28, 33)

                cv2.rectangle(frame, (180, 0), (1100, self.height), (42, 45, 50), -1)
                cv2.line(frame, (180, 0), (180, self.height), (255, 255, 255), 4)
                cv2.line(frame, (1100, 0), (1100, self.height), (255, 255, 255), 4)

                for cx in range(200, 1080, 50):
                    cv2.rectangle(frame, (cx, 360), (cx + 28, 440), (150, 155, 165), -1)

                dash_offset = (dash_offset + 15) % 70
                for y in range(-70 + dash_offset, self.height, 70):
                    cv2.line(frame, (410, y), (410, y + 35), (255, 255, 255), 2)
                    cv2.line(frame, (640, y), (640, y + 35), (0, 220, 255), 3)
                    cv2.line(frame, (870, y), (870, y + 35), (255, 255, 255), 2)

                for v in self.sim_vehicles:
                    v["y"] = float(v["y"] + v["speed_px"])
                    if v["y"] > self.height + 160:
                        v["y"] = float(random.randint(-350, -80))
                        v["speed_px"] = int(random.choice([8, 12, 15, 19, 23]))

                    vx: int = int(v["x"])
                    vy: int = int(v["y"])
                    v_cls: str = v["v_class"]
                    vw, vh = (50, 90) if v_cls == "bike" else (90, 140) if v_cls == "car" else (100, 180)
                    v_color: Tuple[int, int, int] = v["color"]

                    cv2.rectangle(frame, (vx - 2, vy - 2), (vx + vw + 2, vy + vh + 2), (10, 10, 10), -1)
                    cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), v_color, -1)
                    cv2.putText(frame, str(v["plate"][:6]), (vx + 5, vy + vh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (255, 255, 255), 1)

                for p in self.sim_crossings:
                    p["x"] = float(p["x"] + p["vx"])
                    if p["x"] > 1140.0 or p["x"] < 130.0:
                        p["vx"] = -p["vx"]

                    px, py = int(p["x"]), int(p["y"])
                    p_color: Tuple[int, int, int] = (0, 220, 255) if p["c_class"] == "person" else (60, 200, 60)
                    cv2.circle(frame, (px + 15, py + 15), 16, p_color, -1)
                    cv2.rectangle(frame, (px, py), (px + 30, py + 55), p_color, -1)
                    cv2.putText(frame, str(p["label"]), (px - 10, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 1)

                for obj in self.sim_surroundings:
                    ox, oy, ow, oh = int(obj["x"]), int(obj["y"]), int(obj["w"]), int(obj["h"])
                    cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (120, 120, 120), -1)
                    cv2.putText(frame, str(obj["class_name"]), (ox - 10, oy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                                (200, 200, 200), 1)

                cv2.putText(frame, f"AI ITMS RADAR | SURVEILLANCE CORRIDOR | {datetime.now().strftime('%H:%M:%S')}",
                            (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
                time.sleep(0.033)

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            self.frame_queue.put(frame)

    def get_frame(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None


# ===========================================================================
# 14. Velocity & Collision Incident Detector
# ===========================================================================

class VelocityCollisionIncidentDetector:
    """Analyzes trajectories to detect stalled vehicles, wrong-way movement, and violent trajectory overlap."""

    def __init__(self):
        self.active_incidents: Dict[str, float] = {}

    def evaluate_trajectories(self, trajectories: Dict[int, Deque[Tuple[int, int]]],
                              calculated_speeds: Dict[int, float],
                              boxes_dict: Dict[int, np.ndarray]) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []
        now = time.time()

        for track_id, pts in trajectories.items():
            if len(pts) < 6:
                continue

            dy = int(pts[-1][1]) - int(pts[0][1])
            speed = calculated_speeds.get(track_id, 30.0)

            if speed < 3.0:
                key = f"stall_{track_id}"
                if key not in self.active_incidents or (now - self.active_incidents[key] > 15.0):
                    self.active_incidents[key] = now
                    alerts.append({
                        "type": "stalled_vehicle",
                        "track_id": track_id,
                        "desc": f"Vehicle ID #{track_id} STALLED in active corridor"
                    })

            if dy < -40:
                key = f"wrongway_{track_id}"
                if key not in self.active_incidents or (now - self.active_incidents[key] > 15.0):
                    self.active_incidents[key] = now
                    alerts.append({
                        "type": "wrong_side",
                        "track_id": track_id,
                        "desc": f"CRITICAL: Wrong-Way Driver Detected (ID #{track_id})"
                    })

        t_ids = list(boxes_dict.keys())
        for i in range(len(t_ids)):
            for j in range(i + 1, len(t_ids)):
                id_a, id_b = t_ids[i], t_ids[j]
                b_a, b_b = boxes_dict[id_a], boxes_dict[id_b]

                ix1, iy1 = max(int(b_a[0]), int(b_b[0])), max(int(b_a[1]), int(b_b[1]))
                ix2, iy2 = min(int(b_a[2]), int(b_b[2])), min(int(b_a[3]), int(b_b[3]))

                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    area_a = (int(b_a[2]) - int(b_a[0])) * (int(b_a[3]) - int(b_a[1]))
                    area_b = (int(b_b[2]) - int(b_b[0])) * (int(b_b[3]) - int(b_b[1]))
                    denom = float(area_a + area_b - inter_area)
                    if denom > 0:
                        iou = inter_area / denom
                        if iou > 0.45:
                            key = f"collision_{min(id_a, id_b)}_{max(id_a, id_b)}"
                            if key not in self.active_incidents or (now - self.active_incidents[key] > 20.0):
                                self.active_incidents[key] = now
                                alerts.append({
                                    "type": "collision_incident",
                                    "track_id": id_a,
                                    "desc": f"COLLISION ALERT: Vehicles #{id_a} and #{id_b} overlapping"
                                })

        return alerts


# ===========================================================================
# 15. Extended Subsystem Engines (HSRP, EVPP, Weather & Carbon Auditor)
# ===========================================================================

class HSRPPlateComplianceEngine:
    """
    Validates license plates against MoRTH Indian High Security Registration Plate
    (HSRP) syntax standards and flags counterfeit or malformed formats.
    """
    HSRP_REGEX = re.compile(r"^(?P<state>[A-Z]{2})\s?(?P<district>\d{2})\s?(?P<series>[A-Z]{1,3})\s?(?P<number>\d{4})$")

    @classmethod
    def audit_plate_format(cls, plate: str) -> Dict[str, Any]:
        clean_plate = plate.strip().upper()
        match = cls.HSRP_REGEX.match(clean_plate)

        if not match:
            return {
                "plate": clean_plate,
                "is_hsrp_compliant": False,
                "reason": "Malformed or non-standard format (fails MoRTH regex)",
                "flag_for_inspection": True
            }

        groups = match.groupdict()
        state_code = groups["state"]
        district_num = int(groups["district"])

        if state_code == "AP" and not (2 <= district_num <= 42):
            return {
                "plate": clean_plate,
                "is_hsrp_compliant": False,
                "reason": f"Invalid AP District RTO code: AP-{district_num:02d}",
                "flag_for_inspection": True
            }

        return {
            "plate": clean_plate,
            "is_hsrp_compliant": True,
            "state_code": state_code,
            "district_code": f"{state_code} {district_num:02d}",
            "series": groups["series"],
            "registration_number": groups["number"],
            "flag_for_inspection": False
        }


class EmergencyVehiclePreemptionEngine:
    """
    Detects active emergency response units (Ambulance / Fire Services)
    and institutes an automated Green Corridor via signal overrides.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.green_corridor_active: bool = False
        self.emergency_type: Optional[str] = None
        self.corridor_start_time: float = 0.0
        self.active_emergency_vehicle: Optional[str] = None
        self.corridor_duration: float = 30.0

    def trigger_green_corridor(self, emergency_type: str = "AMBULANCE", vehicle_id: Optional[str] = None) -> Dict[
        str, Any]:
        with self.lock:
            self.green_corridor_active = True
            self.emergency_type = emergency_type
            self.corridor_start_time = time.time()
            self.active_emergency_vehicle = vehicle_id or f"EMERGENCY-{int(time.time()) % 10000}"
            smart_signal.set_mode(adaptive=False, manual_phase="GREEN")
            return {
                "status": "GREEN_CORRIDOR_ENGAGED",
                "emergency_type": self.emergency_type,
                "vehicle_id": self.active_emergency_vehicle,
                "preemption_phase": "GREEN",
                "corridor_timeout_sec": self.corridor_duration
            }

    def evaluate_cycle(self) -> Dict[str, Any]:
        with self.lock:
            if not self.green_corridor_active:
                return {"active": False}

            elapsed = time.time() - self.corridor_start_time
            if elapsed >= self.corridor_duration:
                self.green_corridor_active = False
                self.emergency_type = None
                self.active_emergency_vehicle = None
                smart_signal.set_mode(adaptive=True)
                return {"active": False, "status": "EXPIRED_RESTORED_TO_NORMAL"}

            return {
                "active": True,
                "emergency_type": self.emergency_type,
                "vehicle_id": self.active_emergency_vehicle,
                "time_remaining_sec": round(self.corridor_duration - elapsed, 1)
            }

    def cancel_green_corridor(self) -> Dict[str, Any]:
        with self.lock:
            self.green_corridor_active = False
            self.emergency_type = None
            self.active_emergency_vehicle = None
            smart_signal.set_mode(adaptive=True)
            return {"status": "GREEN_CORRIDOR_DEACTIVATED"}


class EnvironmentalWeatherAnalyzer:
    """
    Analyzes frame contrast and darkness index to estimate fog, rain, or glare,
    dynamically computing a safe, enforceable advisory speed limit.
    """

    def __init__(self, base_speed_limit: float = 60.0):
        self.base_speed_limit = base_speed_limit
        self.lock = threading.Lock()
        self.current_visibility_pct: float = 95.0
        self.weather_condition: str = "Clear Sky / Good Visibility"
        self.advisory_speed_kmh: float = base_speed_limit

    def analyze_frame_visibility(self, bgr_frame: np.ndarray) -> Dict[str, Any]:
        if bgr_frame is None or bgr_frame.size == 0:
            return {"visibility_pct": 100.0, "condition": "Clear", "advisory_speed": self.base_speed_limit}

        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        contrast = float(np.std(gray))
        mean_brightness = float(np.mean(gray))

        visibility = max(10.0, min(100.0, (contrast / 65.0) * 100.0))
        condition = "Clear Sky / Good Visibility"
        safe_limit = self.base_speed_limit

        if visibility < 35.0:
            condition = "Dense Fog / Low Visibility Warning"
            safe_limit = max(30.0, self.base_speed_limit - 25.0)
        elif visibility < 55.0:
            condition = "Moderate Rain / Hazy Highway"
            safe_limit = max(40.0, self.base_speed_limit - 15.0)
        elif mean_brightness < 35.0:
            condition = "Nighttime / Low Illumination"
            safe_limit = max(45.0, self.base_speed_limit - 10.0)

        with self.lock:
            self.current_visibility_pct = round(visibility, 1)
            self.weather_condition = condition
            self.advisory_speed_kmh = round(safe_limit, 0)

            return {
                "visibility_pct": self.current_visibility_pct,
                "mean_brightness": round(mean_brightness, 1),
                "weather_condition": self.weather_condition,
                "base_speed_limit": self.base_speed_limit,
                "advisory_speed_kmh": self.advisory_speed_kmh
            }


class FleetCarbonEmissionAuditor:
    """
    Computes real-time CO2 emissions from monitored vehicles based on vehicle class,
    speed, and fuel type. Flags vehicles with expired PUC certificates.
    """
    BASE_EMISSION_FACTORS: Dict[str, float] = {
        "Petrol": 142.0,
        "Diesel": 171.0,
        "CNG": 110.0,
        "Electric": 0.0
    }

    CLASS_MULTIPLIERS: Dict[str, float] = {
        "Two-Wheeler": 0.45,
        "Scooter": 0.40,
        "Cruiser Bike": 0.55,
        "LMV": 1.0,
        "LMV (SUV)": 1.25,
        "LMV (MPV)": 1.20,
        "Heavy Passenger Vehicle": 3.80,
        "Heavy Goods Vehicle": 4.50,
        "Medium Goods Vehicle": 3.10
    }

    def __init__(self):
        self.lock = threading.Lock()
        self.total_co2_grams: float = 0.0
        self.audited_vehicles_count: int = 0
        self.puc_violation_count: int = 0

    def compute_vehicle_emission(self, profile: Dict[str, Any], speed_kmh: float, distance_traveled_m: float = 100.0) -> \
            Dict[str, Any]:
        fuel_type = profile.get("fuel_type", "Petrol")
        veh_class = profile.get("veh_class", "LMV")
        puc_status = profile.get("puc_status", "VALID")

        base_rate = self.BASE_EMISSION_FACTORS.get(fuel_type, 145.0)
        mult = self.CLASS_MULTIPLIERS.get(veh_class, 1.0)

        speed_factor = 1.0
        if speed_kmh > 80.0:
            speed_factor += min(0.4, (speed_kmh - 80.0) * 0.01)
        elif 1.0 < speed_kmh < 15.0:
            speed_factor += 0.35

        emissions_per_km = base_rate * mult * speed_factor
        trip_emissions_g = (distance_traveled_m / 1000.0) * emissions_per_km

        with self.lock:
            self.total_co2_grams += trip_emissions_g
            self.audited_vehicles_count += 1
            if puc_status == "EXPIRED":
                self.puc_violation_count += 1

            return {
                "plate": profile.get("plate", "UNKNOWN"),
                "fuel_type": fuel_type,
                "veh_class": veh_class,
                "co2_rate_g_per_km": round(emissions_per_km, 1),
                "sample_trip_co2_grams": round(trip_emissions_g, 2),
                "puc_compliant": (puc_status == "VALID"),
                "green_tax_eligible": (puc_status == "EXPIRED" or trip_emissions_g > 200.0)
            }

    def get_corridor_emission_summary(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "total_co2_kg": round(self.total_co2_grams / 1000.0, 2),
                "total_vehicles_audited": self.audited_vehicles_count,
                "expired_puc_count": self.puc_violation_count
            }


hsrp_engine = HSRPPlateComplianceEngine()
evpp_engine = EmergencyVehiclePreemptionEngine()
weather_engine = EnvironmentalWeatherAnalyzer(base_speed_limit=60.0)
emission_auditor = FleetCarbonEmissionAuditor()


# ===========================================================================
# 16. Core AI Radar & Safety Processing Engine
# ===========================================================================
