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

For a local PostgreSQL server, create an empty database named
`traffic_safety` (for example, from pgAdmin's Query Tool):

```sql
CREATE DATABASE traffic_safety;
```

Copy `.env.example` to `.env` if you have not already, then set
`DATABASE_URL` in `.env` to your local PostgreSQL URL, using your PostgreSQL
username and URL-encoded password:

```text
DATABASE_URL=postgresql://postgres:YOUR_URL_ENCODED_PASSWORD@localhost:5432/traffic_safety
```

Replace the example username if needed. Keep the password only in `.env`;
never commit or share that file. Start the FastAPI application after setting
the URL; it creates the application's tables in the empty PostgreSQL database.
This fresh-database setup does not import records from the existing SQLite
files, which remain unchanged.

For Streamlit Community Cloud, add `DATABASE_URL` in the app's **Settings →
Secrets** with the same connection URL. The PostgreSQL server must allow
connections from the deployed app.

## Prepare and train a traffic detector

### Build India-representative vehicle data

There is no single dataset in this project that provides labeled vehicle
images from every Indian state. The
[India Driving Dataset (IDD)](https://idd.insaan.iiit.ac.in/) is a useful
starting point: it has 10,000 road-scene images from 182 drives in Hyderabad,
Bengaluru, and nearby areas, with Indian-road labels including auto-rickshaws.
It is not nationwide, and its published annotations are segmentation labels,
not the YOLO bounding boxes expected by this pipeline. Create or convert
instance-level boxes and inspect them before training. Register to access the
dataset and check the license shown on the download page before using or
redistributing it.

For wider coverage, add images and video frames collected or licensed from
different states, cities, road types, cameras, and conditions (day/night,
weather, traffic density). Annotate every target vehicle with a tight bounding
box, including partially occluded and small vehicles. Record each source's
location, capture session, and usage permission. Check for duplicate frames
and keep frames from the same video or drive together when making evaluation
splits; otherwise near-identical frames can make test results look better
than real-world performance. The optional `--group-csv` argument below keeps
each named group in exactly one split. Use a capture-session ID for sequence
leakage checks; use location groups for a separate evaluation of performance
in unseen locations. Group-based split sizes are approximate, especially when
groups have very different numbers of images. Include enough examples of less
common classes, especially tractors and e-rickshaws, rather than relying on
overall image count alone.

Start with this consistent target label order:

1. `car`
2. `motorcycle`
3. `bus`
4. `truck`
5. `bicycle`
6. `auto_rickshaw`
7. `tractor`
8. `e_rickshaw`

Every source's annotations must be mapped to these same IDs before combining
datasets. Do not map ambiguous or missing labels to a target class. This
project's starter YOLO weights can provide a transfer-learning baseline, but
they do not supply India-wide training data or guarantee accurate detections.
Measure per-class precision, recall, and mAP on held-out locations and
conditions, then collect and annotate more examples for classes or regions
that perform poorly.

### Prepare labels and train

Acquire traffic images or extract frames from videos under the source dataset's
terms, then annotate bounding boxes in YOLO detection format. Keep image and
label paths aligned under `images/` and `labels/`; each image needs a matching
`.txt` file (an empty label file represents a negative image). Label values
must be `class_id x_center y_center width height`, with normalized coordinates.
Class IDs follow the order supplied to the preparation command. For the eight
vehicle classes above, use:

For example, given `datasets/source/images/` and `datasets/source/labels/`:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_dataset.py `
  --source datasets/source `
  --output datasets/traffic `
  --classes car motorcycle bus truck bicycle auto_rickshaw tractor e_rickshaw
```

For group-aware evaluation, create `datasets/source/groups.csv` with one row
per image. `image` is relative to `datasets/source/images/`; `group` is the
capture-session or location identifier. Every image must appear exactly once,
and at least three distinct groups are needed:

```csv
image,group
drive_001/frame_000001.jpg,drive_001
drive_001/frame_000002.jpg,drive_001
drive_002/frame_000001.jpg,drive_002
```

Then pass `--group-csv datasets/source/groups.csv` to the preparation command.
Without it, the script uses a reproducible random image-level split.

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

## Run tests

Run the automated test suite with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

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
