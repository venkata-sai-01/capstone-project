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
as the local CSV. Video files, YOLO model weights, and evidence images remain
on disk rather than being stored as large database blobs.

## Configure the YOLO26 detector

The traffic pipeline runs Ultralytics YOLO26 directly and defaults to the
COCO-pretrained nano checkpoint at `assets/models/yolo26n.pt`. Install project
dependencies, then download the official checkpoint into that location:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; from pathlib import Path; p=Path('assets/models/yolo26n.pt'); p.parent.mkdir(parents=True, exist_ok=True); YOLO('yolo26n.pt').save(str(p))"
```

Alternatively, run the app with internet access and Ultralytics can download
the default checkpoint when it is first loaded. Set `YOLO_MODEL_PATH` in `.env`
or pass `--model` to select another Ultralytics detection checkpoint. Model
weights are local runtime files and are not included in Git. If upgrading an
existing `.env` from the ONNX version, change `YOLO_MODEL_PATH` to
`assets/models/yolo26n.pt`.

The detector uses CPU inference by default, full-frame plus overlapping crop
passes for larger frames, class-aware NMS to merge crop results, and the
project's IoU tracker. YOLO26's standard one-to-many detection path is used for
accuracy. Live detections are drawn with class-colored boxes and
class/confidence labels. The COCO checkpoint detects general classes such as
people, cars, and trucks; number-plate OCR and seat-belt classification require
separately trained compatible model weights and are not provided by the
included model.

## Indian vehicle registration codes

The local vehicle-code library recognizes the current registration prefixes
for all 28 states and 8 union territories, legacy prefixes, and RTO/district
registration codes. Lookups are available through
`src.services.india_vehicle_codes.lookup_indian_vehicle_code`; registry
profiles also include the parsed state and RTO code.

The RTO directory in `data/india_rto_codes.csv` is based on the community
[India-Codes RTO dataset](https://github.com/kishorek/India-Codes) and is
supplemented for Telangana and Ladakh using the
[RTO district reference list](https://en.wikipedia.org/wiki/List_of_RTO_districts_in_India).
The Telangana and Ladakh rows are adapted from the Wikipedia reference under
the [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/);
the attribution and source link are retained here.
The directory may be incomplete or out of date, some RTO codes cover multiple
areas, and it is not an official government or VAHAN lookup. Treat a match as
registration-prefix identification only; verify current jurisdiction with the
relevant transport department before using it for enforcement or legal
decisions. Vehicle owner and status details in the local registry remain
simulated unless connected to an authorized data source.

## Evaluate and compare YOLO models

Prepare a labeled evaluation split using standard YOLO detection labels: keep
the image and label trees mirrored, with one `.txt` file per image and rows in
`class_id x_center y_center width height` normalized to `[0, 1]`. Use an empty
label file for a negative image. For example:

```text
datasets/traffic/images/val/highway/frame001.jpg
datasets/traffic/labels/val/highway/frame001.txt
```

Run the same images and labels through one or more YOLO checkpoints:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_yolo `
  --images datasets/traffic/images/val `
  --labels datasets/traffic/labels/val `
  --model assets/models/yolo26n.pt `
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
├── assets/models/           Local YOLO26 checkpoint
├── scripts/                 PostgreSQL provisioning and data migration
├── dashboard/               Read-only Streamlit records viewer
├── data/                    SQLite sources and violation CSV (migration inputs)
│   └── india_rto_codes.csv  State/UT and RTO registration-code reference
├── evidence/violations/     Captured violation snapshots
├── src/
│   ├── main.py              FastAPI entry point, lifecycle, and routes
│   ├── config.py            Environment, paths, and database configuration
│   ├── services/
│   │   ├── integrations.py  Traffic engines and data integrations
│   │   ├── india_vehicle_codes.py  State/UT and RTO registration-code lookup
│   │   ├── pipeline.py      Video capture, inference, and safety analysis
│   │   ├── traffic.py       Compatibility exports for traffic services
│   │   └── quantum.py       Experimental optimization and classification
│   ├── web/
│   │   └── pages.py         Dashboard and API-explorer HTML
│   ├── yolo_detector.py     YOLO inference and tracking
│   └── detection_overlay.py Detection visualization
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
