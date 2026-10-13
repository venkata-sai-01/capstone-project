# Traffic safety system

## Open this project in VS Code

Open `C:\project\CAPSTONE PROJECT` as the workspace folder. The workspace should
show the application source, Streamlit dashboard, data, model, uploads, and
evidence folders in Explorer.

## Set up from Windows Terminal

```powershell
cd "C:\project\CAPSTONE PROJECT"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

The `.env` file is optional for the default local SQLite configuration. Do not
put database passwords or other secrets in source files.

## Configure local PostgreSQL

The project can provision a dedicated `traffic_app` role and `traffic_safety`
database on the local PostgreSQL service. Run the setup utility from a terminal;
it prompts for the PostgreSQL administrator password without echoing it, creates
a generated app password, verifies the connection, and writes the URL only to
the ignored local `.env` file:

```powershell
.\.venv\Scripts\python.exe scripts\configure_postgres.py
```

The setup utility refuses to overwrite an existing role or database. For
different names or a non-default local server, use `--host`, `--port`,
`--admin-user`, `--database`, and `--app-user`.

To copy existing records from the local SQLite stores and violation CSV into
PostgreSQL, run:

```powershell
.\.venv\Scripts\python.exe scripts\migrate_sqlite_to_postgres.py
```

The migration is repeatable: rows with existing primary/unique keys are left
unchanged, and the SQLite files and CSV are retained as a backup. Once
`DATABASE_URL` is configured, both the FastAPI app and the Streamlit records
viewer use PostgreSQL; new violation records are written to PostgreSQL as well
as the local CSV. Video files, ONNX model weights, and evidence images remain
on disk rather than being stored as large database blobs.

## Configure the YOLO ONNX detector

The traffic pipeline uses ONNX Runtime and defaults to the existing COCO-trained
YOLOv8 nano model at `assets/models/yolov8n.onnx` for backward compatibility.
It also accepts compatible exports from newer Ultralytics models, including
YOLO26. For frames larger than the model input, the detector runs a full-frame
pass plus overlapping crop passes and merges their results with class-aware
NMS; this can recover smaller, distant objects at the cost of inference speed.
Model files are not included in Git.

To use the latest YOLO26 nano detection model, install the current Ultralytics
package in an environment with PyTorch, export the model to ONNX, then set
`YOLO_MODEL_PATH=assets/models/yolo26n.onnx` in `.env`:

```powershell
python -m pip install --upgrade ultralytics
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt').export(format='onnx', imgsz=640)"
```

The export writes `yolo26n.onnx` beside the checkpoint (normally in the current
working directory); move it to `assets/models/yolo26n.onnx` or set
`YOLO_MODEL_PATH` to its actual location. The runtime expects NCHW RGB input
and supports standard YOLO detection tensors as well as end-to-end
`xyxy, confidence, class` output rows. PyTorch and Ultralytics are needed only
to obtain/export weights; the application itself continues to use ONNX Runtime.
Live detections are drawn with class-colored boxes and class/confidence labels.
The included COCO model detects general classes such as people, cars, and
trucks; number-plate OCR and seat-belt classification require separately
trained compatible model weights and are not provided by the included model.

## Evaluate and compare YOLO models

Prepare a labeled evaluation split using standard YOLO detection labels: keep
the image and label trees mirrored, with one `.txt` file per image and rows in
`class_id x_center y_center width height` normalized to `[0, 1]`. Use an empty
label file for a negative image. For example:

```text
datasets/traffic/images/val/highway/frame001.jpg
datasets/traffic/labels/val/highway/frame001.txt
```

Run the same images and labels through one or more ONNX models:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_yolo `
  --images datasets/traffic/images/val `
  --labels datasets/traffic/labels/val `
  --model assets/models/yolov8n.onnx `
  --model assets/models/yolo26n.onnx `
  --output reports/yolo-evaluation.json
```

The report includes precision, recall, F1 at confidence `0.25`, mAP@0.5,
mAP@0.5:0.95, per-class AP, false positives/negatives, model-load time, and
mean/median/p95 inference latency. The evaluator uses a `0.001` minimum
inference confidence by default for the mAP confidence curve; tune it with
`--min-confidence` if needed, and use `--metric-confidence` to change the
precision/recall operating point. Use the same held-out labeled split for every
model; the project does not include a labeled traffic evaluation dataset, so
metrics are only as representative as the images and annotations you provide.

## Run the applications

Run the main FastAPI traffic safety system:

```powershell
.\.venv\Scripts\python.exe -m src.main
```

It prints the web dashboard, API explorer, and REST API URLs in the terminal.
Use `Ctrl+C` in that terminal to stop it.

The separate read-only Streamlit viewer for the local database and violation
records can be started with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

In VS Code, use **Terminal → Run Task** and choose either app task.

## Project layout

```text
CAPSTONE PROJECT/
├── .vscode/                 VS Code interpreter and terminal tasks
├── assets/models/           Local YOLO ONNX model
├── scripts/                 PostgreSQL provisioning and data migration
├── dashboard/               Read-only Streamlit records viewer
├── data/                    SQLite sources and violation CSV (migration inputs)
├── evidence/violations/     Captured violation snapshots
├── src/                     FastAPI application and traffic pipeline
├── uploads/
│   ├── citizen_uploads/     Citizen report attachments
│   └── videos/              Uploaded traffic videos
├── tests/                   Automated tests
├── ARCHITECTURE.md          System components, runtime/data flow, and limits
├── .env.example             Local configuration template
├── requirements.txt         Python dependencies
└── README.md
```

The primary FastAPI application uses paths relative to this project root, so
database files, model weights, uploads, and evidence stay in their respective
folders even when it is launched with `python -m src.main`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the component diagram, video
inference flow, persistence model, service interfaces, and known limitations.
