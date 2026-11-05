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
import threading
import time
import uuid
import math
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
if __package__:
    from src.auth import (
        AuthenticationConfigurationError,
        OPERATOR,
        PUBLIC_HTTP_PATHS,
        authenticate_token,
        required_role,
        role_allows,
    )
    from src.detection_overlay import detection_color, draw_detection_overlay
    from src.yolo_detector import YOLO26Detector
else:
    from auth import (
        AuthenticationConfigurationError,
        OPERATOR,
        PUBLIC_HTTP_PATHS,
        authenticate_token,
        required_role,
        role_allows,
    )
    from detection_overlay import detection_color, draw_detection_overlay
    from yolo_detector import YOLO26Detector


if __package__:
    from src.config import (
        PROJECT_ROOT, DATA_DIR, UPLOADS_DIR, VIDEO_UPLOAD_DIR, CITIZEN_UPLOAD_DIR,
        EVIDENCE_DIR, CSV_LOG_FILE, DEFAULT_MODEL_PATH, DATABASE_URL,
        SQLITE_DATABASE_DIR, DASHBOARD_API_URL, DASHBOARD_UI_URL,
        DASHBOARD_API_INTERFACE_URL, DASHBOARD_ORIGINS, MODEL_PATH,
        _project_path, _load_local_environment, _connect_database,
        _database_backend, _start_auxiliary_server,
    )
    from src.services.traffic import (
        AP_STATUTORY_PENALTIES, OFFICIAL_AP_RTO_MATRIX, FIRST_NAMES, LAST_NAMES,
        FLEET_SPECS, SimVehicle, SimCrossing, SimSurrounding, BlockchainBlock,
        CitizenSafetyReport, EnvironmentalAdaptivityEngine, DynamicROIMasker,
        ChallanBlockchainLedger, PredictiveAccidentPreventionEngine,
        SmartAdaptiveTrafficLightEngine, CitizenSafetyPortalEngine,
        APCCTNSHotlistDatabase, APRTOIntegrationGateway, APVehicleRegistryEngine,
        TimeSeriesMetricsLogger, AcousticAnalysisEngine, IngestionStreamWatchdog,
        DynamicTrafficGrabber, VelocityCollisionIncidentDetector,
        HSRPPlateComplianceEngine, EmergencyVehiclePreemptionEngine,
        EnvironmentalWeatherAnalyzer, FleetCarbonEmissionAuditor,
        APPoliceRadarSafetyEngine, WebRTCVideoStreamTrack, blockchain_ledger,
        predictive_engine, smart_signal, citizen_portal, ap_cctns_radar,
        ap_rto_gateway, vahan_db, timeseries_logger, acoustic_radar, hsrp_engine,
        evpp_engine, weather_engine, emission_auditor,
    )
    from src.services.quantum import (
        GroversVehicleDatabaseSearchEngine, QuantumMLHazardClassifier,
        TrafficQAOAOptimizer, grover_search_engine, qaoa_optimizer,
        qsvm_classifier,
    )
    from src.web.pages import build_api_interface_page, build_dashboard_page
else:
    from config import (
        PROJECT_ROOT, DATA_DIR, UPLOADS_DIR, VIDEO_UPLOAD_DIR, CITIZEN_UPLOAD_DIR,
        EVIDENCE_DIR, CSV_LOG_FILE, DEFAULT_MODEL_PATH, DATABASE_URL,
        SQLITE_DATABASE_DIR, DASHBOARD_API_URL, DASHBOARD_UI_URL,
        DASHBOARD_API_INTERFACE_URL, DASHBOARD_ORIGINS, MODEL_PATH,
        _project_path, _load_local_environment, _connect_database,
        _database_backend, _start_auxiliary_server,
    )
    from services.traffic import (
        AP_STATUTORY_PENALTIES, OFFICIAL_AP_RTO_MATRIX, FIRST_NAMES, LAST_NAMES,
        FLEET_SPECS, SimVehicle, SimCrossing, SimSurrounding, BlockchainBlock,
        CitizenSafetyReport, EnvironmentalAdaptivityEngine, DynamicROIMasker,
        ChallanBlockchainLedger, PredictiveAccidentPreventionEngine,
        SmartAdaptiveTrafficLightEngine, CitizenSafetyPortalEngine,
        APCCTNSHotlistDatabase, APRTOIntegrationGateway, APVehicleRegistryEngine,
        TimeSeriesMetricsLogger, AcousticAnalysisEngine, IngestionStreamWatchdog,
        DynamicTrafficGrabber, VelocityCollisionIncidentDetector,
        HSRPPlateComplianceEngine, EmergencyVehiclePreemptionEngine,
        EnvironmentalWeatherAnalyzer, FleetCarbonEmissionAuditor,
        APPoliceRadarSafetyEngine, WebRTCVideoStreamTrack, blockchain_ledger,
        predictive_engine, smart_signal, citizen_portal, ap_cctns_radar,
        ap_rto_gateway, vahan_db, timeseries_logger, acoustic_radar, hsrp_engine,
        evpp_engine, weather_engine, emission_auditor,
    )
    from services.quantum import (
        GroversVehicleDatabaseSearchEngine, QuantumMLHazardClassifier,
        TrafficQAOAOptimizer, grover_search_engine, qaoa_optimizer,
        qsvm_classifier,
    )
    from web.pages import build_api_interface_page, build_dashboard_page


# ===========================================================================
# 18. FastAPI App Lifecycle, WebSockets & Dashboard Interface
# ===========================================================================

pipeline_grabber: Optional[DynamicTrafficGrabber] = None
pipeline_engine: Optional[APPoliceRadarSafetyEngine] = None
relay = MediaRelay()
peer_connections: Set[RTCPeerConnection] = set()
active_websocket_clients: Set[WebSocket] = set()
master_track: Optional[WebRTCVideoStreamTrack] = None
VIDEO_SRC: Union[int, str] = 0

os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)
os.makedirs(CITIZEN_UPLOAD_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global pipeline_grabber, pipeline_engine, master_track
    grabber = DynamicTrafficGrabber(src=VIDEO_SRC, width=1280, height=720)
    engine = APPoliceRadarSafetyEngine(grabber=grabber, model_path=MODEL_PATH)
    pipeline_grabber = grabber
    pipeline_engine = engine

    grabber.start()
    engine.start()
    acoustic_radar.start()
    ap_rto_gateway.start()

    master_track = WebRTCVideoStreamTrack(engine, fps=30)

    async def send_telemetry(client: WebSocket, message: str) -> bool:
        try:
            await asyncio.wait_for(client.send_text(message), timeout=0.5)
            return True
        except (WebSocketDisconnect, ConnectionResetError, RuntimeError, asyncio.TimeoutError):
            return False

    async def ws_broadcast_loop():
        while True:
            if active_websocket_clients and pipeline_engine is not None:
                _, meta = pipeline_engine.get_latest_frame_and_telemetry()
                msg = json.dumps(meta)
                clients = list(active_websocket_clients)
                results = await asyncio.gather(
                    *(send_telemetry(client, msg) for client in clients)
                )
                active_websocket_clients.difference_update(
                    client for client, sent in zip(clients, results) if not sent
                )
            await asyncio.sleep(0.1)

    broadcast_task = asyncio.create_task(ws_broadcast_loop())
    yield

    broadcast_task.cancel()
    if pipeline_grabber is not None:
        pipeline_grabber.stop()
    if pipeline_engine is not None:
        pipeline_engine.stop()
    acoustic_radar.stop()
    ap_rto_gateway.stop()
    await asyncio.gather(*[pc.close() for pc in peer_connections], return_exceptions=True)
    peer_connections.clear()


app = FastAPI(
    title="Real-Time Multi-Hazard Traffic Surveillance and Automated e-Challan System using Edge YOLO and WebRTC",
    lifespan=lifespan)
dashboard_app = FastAPI(title="Traffic Safety Dashboard", docs_url=None, openapi_url=None)
api_interface_app = FastAPI(title="Traffic Safety API Explorer", docs_url=None, openapi_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=DASHBOARD_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
MAX_VIDEO_UPLOAD_BYTES = 500 * 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4"}
ALLOWED_VIDEO_CONTENT_TYPES = {
    "application/octet-stream",
    "video/mp4",
    "video/quicktime",
    "video/x-matroska",
    "video/x-msvideo",
}


@app.get("/")
async def api_root():
    return {
        "service": "traffic-safety-api",
        "status": "ok",
        "docs": "/docs",
        "health": "/api/health",
        "dashboard": DASHBOARD_UI_URL,
    }


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id
    started_at = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started_at) * 1000:.2f}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path.startswith("/api/") or request.url.path == "/offer":
        response.headers["Cache-Control"] = "no-store"
    return response


@app.middleware("http")
async def api_authentication_middleware(request: Request, call_next):
    path = request.url.path
    if request.method == "OPTIONS" or path in PUBLIC_HTTP_PATHS:
        return await call_next(request)
    if path != "/offer" and not path.startswith("/api/"):
        return await call_next(request)

    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return JSONResponse(
            {"detail": "A bearer access token is required."},
            status_code=401,
            headers={
                "WWW-Authenticate": "Bearer",
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )
    try:
        role = authenticate_token(token.strip())
    except AuthenticationConfigurationError as error:
        return JSONResponse(
            {"detail": str(error)},
            status_code=503,
            headers={"Cache-Control": "no-store"},
        )
    needed_role = required_role(request.method)
    if role is None:
        return JSONResponse(
            {"detail": "The bearer access token is invalid."},
            status_code=401,
            headers={
                "WWW-Authenticate": "Bearer",
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )
    if not role_allows(role, needed_role):
        return JSONResponse(
            {"detail": f"The {needed_role} role is required for this operation."},
            status_code=403,
            headers={"Cache-Control": "no-store"},
        )
    request.state.authenticated_role = role
    return await call_next(request)


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    origin = websocket.headers.get("origin")
    if origin and origin.rstrip("/") not in DASHBOARD_ORIGINS:
        await websocket.close(code=4403, reason="WebSocket origin is not allowed.")
        return
    await websocket.accept()
    try:
        try:
            auth_message = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        except asyncio.TimeoutError:
            await websocket.close(code=4401, reason="Authentication timed out.")
            return
        try:
            auth_data = json.loads(auth_message)
        except json.JSONDecodeError:
            await websocket.close(code=4401, reason="Authentication is required.")
            return
        try:
            role = (
                authenticate_token(auth_data.get("token"))
                if isinstance(auth_data, dict) and auth_data.get("type") == "auth"
                else None
            )
        except AuthenticationConfigurationError:
            await websocket.close(code=1011, reason="Access control is not configured.")
            return
        if role is None:
            await websocket.close(code=4401, reason="Invalid access token.")
            return
        await websocket.send_json({"type": "authenticated", "role": role})
        active_websocket_clients.add(websocket)
        while True:
            data_text = await websocket.receive_text()
            try:
                data = json.loads(data_text)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Telemetry settings must be valid JSON."})
                continue
            if not isinstance(data, dict):
                await websocket.send_json({"error": "Telemetry settings must be a JSON object."})
                continue
            if role != OPERATOR:
                await websocket.send_json({"error": "Operator access is required to change settings."})
                continue
            if pipeline_engine is not None:
                pipeline_engine.update_params(
                    speed_limit=data.get("speed_limit"),
                    distance_m=data.get("distance_m"),
                    conf=data.get("conf"),
                    line1_y=data.get("line1_y"),
                    line2_y=data.get("line2_y")
                )
    except (WebSocketDisconnect, ConnectionResetError, RuntimeError):
        pass
    finally:
        active_websocket_clients.discard(websocket)


START_TIME = time.monotonic()


def _database_health_check() -> Dict[str, str]:
    connection = _connect_database("traffic_registry.db")
    try:
        row = connection.cursor().execute("SELECT 1").fetchone()
        if row != (1,):
            raise RuntimeError(f"Unexpected database health-check response: {row!r}")
    finally:
        connection.close()
    return {"status": "ok", "database": _database_backend()}


def _system_overview() -> Dict[str, Any]:
    base_health = _database_health_check()
    return {
        "project_name": "AI Traffic Safety Command Center",
        "status": "demo_operational",
        "deployment_class": "local_prototype",
        "database": base_health["database"],
        "uptime_seconds": round(time.monotonic() - START_TIME, 2),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ai_layers": [
            {"name": "Vision intelligence", "detail": "Local YOLO inference + ROI and CLAHE preprocessing", "active": True},
            {"name": "Predictive risk engine", "detail": "Local TTC, lane drift, stall and wrong-way heuristics", "active": True},
            {"name": "Signal optimization", "detail": "Local adaptive signal and emergency-preemption simulation", "active": True},
            {"name": "Safety portal", "detail": "Local citizen reports and SOS records; no dispatch integration", "active": True},
            {"name": "Evidence ledger", "detail": "Local SHA-256 linked records; not a distributed blockchain", "active": True},
        ],
        "system_summary": {
            "stream_mode": (
                "simulator"
                if pipeline_grabber is not None and not pipeline_grabber.has_source
                else "configured_source"
            ),
            "adaptive_signal": smart_signal.adaptive_mode,
            "hotlist_status": "local_demo_data",
            "weather_engine": "frame_heuristic",
            "quantum_layer": "experimental_local_simulation",
        },
        "external_integrations": {
            "government_registry": "not_connected_local_demo_data",
            "e_challan_dispatch": ap_rto_gateway.dispatch_status,
            "emergency_services": "not_connected",
            "quantum_hardware": "not_connected",
        },
    }


@app.get("/api/health")
async def api_health():
    return JSONResponse(await asyncio.to_thread(_database_health_check))


@app.get("/api/system/overview")
async def api_system_overview():
    return JSONResponse(_system_overview())


@app.get("/api/detections")
async def api_detections():
    if pipeline_engine is None:
        raise HTTPException(status_code=503, detail="Detection pipeline is not initialized.")
    _, telemetry = pipeline_engine.get_latest_frame_and_telemetry()
    return JSONResponse({
        "model": telemetry.get("detection_model", os.path.basename(MODEL_PATH)),
        "counts": telemetry.get("object_counts", {}),
        "detections": telemetry.get("detections", []),
    })


@api_interface_app.get("/", response_class=HTMLResponse)
async def api_interface():
    return build_api_interface_page(app.openapi(), DASHBOARD_API_URL, DASHBOARD_UI_URL)


@app.get("/api-interface", include_in_schema=False)
async def api_interface_redirect():
    return RedirectResponse(DASHBOARD_API_INTERFACE_URL)


@dashboard_app.get("/", response_class=HTMLResponse)
async def index():
    return build_dashboard_page(DASHBOARD_API_URL, DASHBOARD_API_INTERFACE_URL)


# ===========================================================================
# 19. Endpoints, WebRTC Signaling & Integrated APIs
# ===========================================================================

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    client_offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    peer_connections.add(pc)

    if master_track is None:
        raise HTTPException(status_code=503, detail="WebRTC master track not initialized")

    relayed_track = relay.subscribe(master_track)
    pc.addTrack(relayed_track)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(msg):
            try:
                data = json.loads(msg)
                if pipeline_engine is not None:
                    pipeline_engine.update_params(
                        speed_limit=data.get("speed_limit"),
                        distance_m=data.get("distance_m"),
                        conf=data.get("conf"),
                        line1_y=data.get("line1_y"),
                        line2_y=data.get("line2_y")
                    )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"[DataChannel Error] {e}")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            peer_connections.discard(pc)

    await pc.setRemoteDescription(client_offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.get("/api/vahan_lookup")
async def api_vahan_lookup(query: str):
    records = await asyncio.to_thread(vahan_db.search_records, query)
    return JSONResponse({"status": "found" if records else "not_found", "results": records})


@app.post("/api/upload_video")
async def api_upload_video(file: UploadFile = File(...)):
    extension = os.path.splitext(file.filename or "")[1].lower()
    content_type = (file.content_type or "").lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Upload a supported MP4, AVI, MOV, or MKV video.")
    if content_type and content_type not in ALLOWED_VIDEO_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="The uploaded file must have a supported video content type.")

    file_name = f"{uuid.uuid4().hex}{extension}"
    file_location = os.path.join(VIDEO_UPLOAD_DIR, file_name)
    total_bytes = 0
    try:
        with open(file_location, "xb") as destination:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video uploads are limited to {MAX_VIDEO_UPLOAD_BYTES // (1024 * 1024)} MB.",
                    )
                destination.write(chunk)
        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="The uploaded video is empty.")

        capture = cv2.VideoCapture(file_location)
        try:
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if (
                not capture.isOpened()
                or not math.isfinite(width)
                or not math.isfinite(height)
                or width <= 0
                or height <= 0
                or width > 8192
                or height > 8192
            ):
                raise HTTPException(status_code=415, detail="The uploaded file is not a supported video.")
            decoded, _ = capture.read()
            if not decoded:
                raise HTTPException(status_code=415, detail="The uploaded video contains no decodable frames.")
        finally:
            capture.release()

        if pipeline_grabber is not None:
            pipeline_grabber.switch_source(file_location)
    except Exception:
        if os.path.exists(file_location):
            os.remove(file_location)
        raise
    finally:
        await file.close()
    return JSONResponse({"status": "ok", "filename": file_name})


@app.post("/api/switch_source")
async def api_switch_source(source: str):
    if pipeline_grabber is not None:
        src = 0 if source == "0" else "synthetic_sim"
        pipeline_grabber.switch_source(src)
    return JSONResponse({"status": "ok", "source": source})


@app.get("/api/view_snapshot")
async def view_snapshot(file: str):
    clean_filename = os.path.basename(file)
    file_path = os.path.join(EVIDENCE_DIR, clean_filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg")
    return Response(content="Evidence file not found", status_code=404)


@app.get("/api/export_violations")
async def export_violations():
    csv_file = CSV_LOG_FILE
    if os.path.exists(csv_file):
        return FileResponse(csv_file, media_type="text/csv", filename="Traffic_Violations_Report.csv")
    return Response(content="No violations logged yet", status_code=404)


@app.post("/api/traffic_signal/override")
async def api_signal_override(phase: str):
    smart_signal.set_mode(adaptive=False, manual_phase=phase.upper())
    return JSONResponse({"status": "ok", "phase": phase.upper(), "mode": "MANUAL"})


@app.post("/api/traffic_signal/adaptive_toggle")
async def api_signal_adaptive_toggle():
    smart_signal.set_mode(adaptive=True)
    return JSONResponse({"status": "ok", "mode": "ADAPTIVE_AUTOMATIC"})


@app.get("/api/blockchain/ledger")
async def api_blockchain_ledger():
    def read_ledger():
        blocks = blockchain_ledger.get_latest_blocks(limit=25)
        valid, msg, count = blockchain_ledger.verify_chain_integrity()
        return {"verified": valid, "status": msg, "total_blocks": count, "blocks": blocks}

    return JSONResponse(await asyncio.to_thread(read_ledger))


@app.post("/api/citizen/report_hazard")
async def api_citizen_report(request: Request):
    data = await request.json()
    rep = await asyncio.to_thread(
        citizen_portal.create_report,
        r_type=data.get("report_type", "HAZARD"),
        location=data.get("location", "Highway Corridor"),
        desc=data.get("description", "Citizen Observation"),
        plate=data.get("plate_number"),
        phone=data.get("contact_phone")
    )
    return JSONResponse({"status": "recorded", "report": rep})


@app.post("/api/citizen/sos_alert")
async def api_citizen_sos(request: Request):
    data = await request.json()
    sos_res = await asyncio.to_thread(
        citizen_portal.trigger_sos,
        location=data.get("location", "Live Geo GPS"),
        details=data.get("details", "Citizen SOS Triggered")
    )
    return JSONResponse(sos_res)


@app.get("/api/citizen/list_reports")
async def api_citizen_list_reports():
    reports = await asyncio.to_thread(citizen_portal.get_recent_reports, limit=20)
    return JSONResponse({"reports": reports})


# --- Feature Extension Endpoints ---

@app.get("/api/hsrp_check")
async def api_hsrp_check(plate: str):
    result = hsrp_engine.audit_plate_format(plate)
    return JSONResponse(result)


@app.post("/api/green_corridor/trigger")
async def api_trigger_green_corridor(request: Request):
    data = await request.json()
    e_type = data.get("emergency_type", "AMBULANCE")
    v_id = data.get("vehicle_id")
    res = evpp_engine.trigger_green_corridor(emergency_type=e_type, vehicle_id=v_id)
    return JSONResponse(res)


@app.post("/api/green_corridor/cancel")
async def api_cancel_green_corridor():
    res = evpp_engine.cancel_green_corridor()
    return JSONResponse(res)


@app.get("/api/green_corridor/status")
async def api_green_corridor_status():
    return JSONResponse(evpp_engine.evaluate_cycle())


@app.get("/api/weather_advisory")
async def api_weather_advisory():
    return JSONResponse({
        "visibility_pct": weather_engine.current_visibility_pct,
        "condition": weather_engine.weather_condition,
        "advisory_speed_kmh": weather_engine.advisory_speed_kmh
    })


@app.get("/api/carbon_audit")
async def api_carbon_audit():
    return JSONResponse(emission_auditor.get_corridor_emission_summary())


# ===========================================================================
# 21. Quantum REST API Endpoints
# ===========================================================================

@app.post("/api/quantum/qaoa_optimize")
async def api_quantum_qaoa_optimize(request: Request):
    """Runs a local statevector optimization demo; it does not control traffic signals."""
    try:
        data = await request.json()
        congestion = data.get("corridor_congestion", [4.5, 7.8, 3.2, 8.9])
        p_layers = data.get("p_layers", 2)
        opt = TrafficQAOAOptimizer(p_layers=p_layers)
        result = await asyncio.to_thread(opt.optimize_corridor_flow, congestion_weights=congestion)
        return JSONResponse(result)
    except Exception as e:
        print(f"Error in /api/quantum/qaoa_optimize: {e}")
        return JSONResponse({"status": "error", "message": "An internal error occurred."}, status_code=500)


@app.post("/api/quantum/grover_search")
async def api_quantum_grover_search(request: Request):
    """Runs a local classical statevector search demo against bundled registry data."""
    try:
        data = await request.json()
        target_plate = data.get("target_plate", "AP 03 BK 4921")
        result = await asyncio.to_thread(grover_search_engine.quantum_search_plate, target_plate=target_plate)
        return JSONResponse(result)
    except Exception as e:
        print(f"Error in /api/quantum/grover_search: {e}")
        return JSONResponse({"status": "error", "message": "An internal error occurred."}, status_code=500)


@app.post("/api/quantum/qsvm_hazard_classify")
async def api_quantum_qsvm_classify(request: Request):
    """Runs an experimental local heuristic on telemetry; it is not validated for enforcement."""
    try:
        data = await request.json()
        spd = float(data.get("speed_kmh", 85.0))
        drift = float(data.get("lateral_drift_px", 42.0))
        gap = float(data.get("gap_m", 8.0))
        db = float(data.get("sound_db", 92.0))
        result = await asyncio.to_thread(
            qsvm_classifier.classify_hazard,
            speed_kmh=spd,
            lateral_drift_px=drift,
            gap_m=gap,
            sound_db=db
        )
        return JSONResponse(result)
    except Exception as e:
        print(f"Error in /api/quantum/qsvm_hazard_classify: {e}")
        return JSONResponse({"status": "error", "message": "An internal error occurred."}, status_code=500)


@app.get("/api/quantum/status")
async def api_quantum_status():
    """Returns the operational status of all three quantum computing modules."""
    return JSONResponse({
        "status": "EXPERIMENTAL_LOCAL_SIMULATION",
        "quantum_hardware_connected": False,
        "production_validated": False,
        "modules": {
            "QAOA": "Experimental local optimization simulation; no quantum hardware connection",
            "Grover_Search": "Experimental local classical search demonstration; no claimed quadratic speedup",
            "QSVM_HHL": "Experimental local heuristic; not a validated quantum classifier or HHL deployment",
        }
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-Time Multi-Hazard Traffic Surveillance and Automated e-Challan System using Edge YOLO and WebRTC",
        epilog=(
            "Set DATABASE_URL in the local .env file to use PostgreSQL. SQLite is used "
            "when it is unset. Configure DASHBOARD_API_URL, DASHBOARD_API_INTERFACE_URL, "
            "and DASHBOARD_ORIGINS when the API and web pages are accessed from different hosts."
        ))
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: camera index (0), RTSP URL, or file path")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Ultralytics YOLO26 .pt model path")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP address")
    parser.add_argument("--port", type=int, default=8000, help="REST API port")
    parser.add_argument("--dashboard-port", type=int, default=8080, help="Dashboard web port")
    parser.add_argument("--api-interface-port", type=int, default=8081, help="API Explorer web port")
    parser.add_argument("--env", type=str, default="SANDBOX", choices=["SANDBOX", "PRODUCTION"],
                        help="RTO Server Environment")
    args = parser.parse_args()

    if not DASHBOARD_API_URL:
        DASHBOARD_API_URL = f"http://127.0.0.1:{args.port}"
    if "DASHBOARD_UI_URL" not in os.environ:
        DASHBOARD_UI_URL = f"http://127.0.0.1:{args.dashboard_port}"
    if "DASHBOARD_API_INTERFACE_URL" not in os.environ:
        DASHBOARD_API_INTERFACE_URL = f"http://127.0.0.1:{args.api_interface_port}"
    if DASHBOARD_API_INTERFACE_URL not in DASHBOARD_ORIGINS:
        DASHBOARD_ORIGINS.append(DASHBOARD_API_INTERFACE_URL)
    if len({args.port, args.dashboard_port, args.api_interface_port}) != 3:
        parser.error("--port, --dashboard-port, and --api-interface-port must all be different.")

    VIDEO_SRC = int(args.source) if args.source.isdigit() else args.source
    MODEL_PATH = args.model if os.path.isabs(args.model) else _project_path(args.model)
    ap_rto_gateway.env = args.env
    ap_rto_gateway.dispatch_status = (
        "sandbox_simulation" if args.env == "SANDBOX" else "configuration_required"
    )

    print("\n" + "=" * 85)
    print(" REAL-TIME MULTI-HAZARD TRAFFIC SURVEILLANCE AND AUTOMATED E-CHALLAN SYSTEM")
    print(f" Web Dashboard:     {DASHBOARD_UI_URL}")
    print(f" API Explorer:      {DASHBOARD_API_INTERFACE_URL}")
    print(f" REST API:          {DASHBOARD_API_URL} (docs: {DASHBOARD_API_URL}/docs)")
    print(
        f" Environment:       {args.env} (RTO dispatch: "
        f"{ap_rto_gateway.dispatch_status.replace('_', ' ')})"
    )
    database_location = (
        "DATABASE_URL" if DATABASE_URL else os.path.abspath(SQLITE_DATABASE_DIR or ".")
    )
    print(f" Database:          {_database_backend().upper()} ({database_location})")
    print(" Hash-Linked Ledger: ACTIVE (local SHA-256 chain; not distributed blockchain)")
    print(" Smart Signals:     ADAPTIVE (Density-Based Green Timing & Preemption)")
    print(" Accident Predict:  LOCAL HEURISTIC DEMO (NOT VALIDATED)")
    print(" Citizen Portal:    LOCAL REPORT/SOS RECORDS (NO EXTERNAL DISPATCH)")
    print(" Green Corridor:    LOCAL PREEMPTION SIMULATION (NO PHYSICAL SIGNALS)")
    print(" Weather Engine:    FRAME-BASED VISIBILITY HEURISTIC")
    print(" Carbon Audit:      LOCAL EMISSION ESTIMATE")
    print(" HSRP Audit:        LOCAL PLATE-FORMAT CHECK ONLY")
    print(" Local Optimizer:   EXPERIMENTAL STATEVECTOR DEMO (NO QUANTUM HARDWARE)")
    print(" Local Search:      CLASSICAL DEMO (NO QUANTUM SPEEDUP)")
    print(" Hazard Heuristic:  UNVALIDATED LOCAL DEMO (NOT FOR ENFORCEMENT)")
    print("=" * 85 + "\n")
    _start_auxiliary_server(dashboard_app, "Dashboard", args.host, args.dashboard_port)
    _start_auxiliary_server(api_interface_app, "API Explorer", args.host, args.api_interface_port)
    uvicorn.run(app, host=args.host, port=args.port)