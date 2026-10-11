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

## Connect PostgreSQL

Create a PostgreSQL database, then set its connection URL in the project-root
`.env` file:

```text
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DATABASE
```

Replace each placeholder with the values from your PostgreSQL provider.
URL-encode special characters in the username or password and keep `.env`
private; it is excluded from Git. With `DATABASE_URL` set, both the FastAPI
service and Streamlit records dashboard use PostgreSQL. The dashboard expects
the application tables (`blockchain_blocks`, `citizen_reports`,
`vahan_registry`, and `highway_telemetry`) in that database; starting the
FastAPI service creates the tables. If `DATABASE_URL` is unset, the
applications continue to use the local SQLite files.

For Streamlit Community Cloud, add `DATABASE_URL` in the app's **Settings →
Secrets** with the same connection URL. The PostgreSQL server must allow
connections from the deployed app.

## Prepare and train a traffic detector

Acquire traffic images or extract frames from videos under the source dataset's
terms, then annotate bounding boxes in YOLO detection format. Keep image and
label paths aligned under `images/` and `labels/`; each image needs a matching
`.txt` file (an empty label file represents a negative image). Label values
must be `class_id x_center y_center width height`, with normalized coordinates.
Class IDs follow the order supplied to the preparation command.

For example, given `datasets/source/images/` and `datasets/source/labels/`:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_dataset.py `
  --source datasets/source `
  --output datasets/traffic `
  --classes car motorcycle bus truck person helmet traffic_light
```

The preparation script validates annotations and creates reproducible train,
validation, and test splits plus `datasets/traffic/dataset.yaml`. It refuses
to overwrite a non-empty output directory. Fine-tune and evaluate on the held
out test split with:

```powershell
.\.venv\Scripts\python.exe scripts/train_detector.py `
  --data datasets/traffic/dataset.yaml `
  --weights yolov8n.pt `
  --epochs 100 `
  --imgsz 640 `
  --device cpu
```

Use `--device 0` for the first CUDA GPU when available. Training writes
`best.pt` and a `test_metrics.json` report (precision, recall, mAP and timing)
under `runs/detect/traffic-safety/`. You can connect the resulting weights to
the existing YOLO/WebRTC pipeline with `--model
runs/detect/traffic-safety/weights/best.pt` when starting `src.main`.
`datasets/` and `runs/` are excluded from Git so large datasets and generated
weights are not published accidentally.

Detection labels are not violation decisions: speeding requires calibrated
tracking over time, and red-light jumping requires signal-state and stop-line
logic. Helmet and signal-jumping challan decisions are not currently driven
by trained model detections in the runtime pipeline; validate and implement
those decision rules before using the system for enforcement.

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
├── assets/models/           YOLO model weights
├── dashboard/               Read-only Streamlit records viewer
├── data/                    SQLite databases and violation CSV
├── evidence/violations/     Captured violation snapshots
├── src/                     FastAPI application and traffic pipeline
├── scripts/                 Dataset preparation and YOLO training utilities
├── uploads/
│   ├── citizen_uploads/     Citizen report attachments
│   └── videos/              Uploaded traffic videos
├── tests/                   Automated tests
├── .env.example             Local configuration template
├── requirements.txt         Python dependencies
└── README.md
```

The primary FastAPI application uses paths relative to this project root, so
database files, model weights, uploads, and evidence stay in their respective
folders even when it is launched with `python -m src.main`.
