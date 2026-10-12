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

## Configure the ONNX detector

The traffic pipeline uses ONNX Runtime and expects a YOLOv8 detection model at
`assets/models/yolov8n.onnx` by default. The default model is the COCO-trained
YOLOv8 nano model. For frames larger than the model input, the detector runs a
full-frame pass plus overlapping crop passes and merges their results with
class-aware NMS; this can recover smaller, distant objects at the cost of
inference speed. Model files are not included in Git.
Provide a compatible ONNX export and set `YOLO_MODEL_PATH` in `.env` if it is
stored elsewhere. The detector expects an NCHW RGB input and a standard YOLOv8
detection output. The previous PyTorch `.pt` checkpoint cannot be loaded
directly by this runtime. If you have the `.pt` checkpoint, export it on a
machine where PyTorch and Ultralytics are permitted:

```powershell
python -c "from ultralytics import YOLO; YOLO('assets/models/yolov8n.pt').export(format='onnx', imgsz=640)"
```

This writes `assets/models/yolov8n.onnx`. The application itself uses ONNX
Runtime and does not require PyTorch or Ultralytics.

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
