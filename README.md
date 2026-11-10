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

## Configure API access control

The REST API protects registry searches, violation/evidence exports, reports,
telemetry, and other non-health endpoints with bearer tokens. Viewer tokens can
read protected data; operator tokens can also change settings, switch sources,
upload video, trigger local simulations, and create reports. Access is denied
when credentials are missing or invalid. Generate separate random tokens in
the ignored local `.env` file:

```powershell
.\.venv\Scripts\python.exe scripts\configure_api_access.py
```

Restart the API process after generating or rotating tokens so it reloads the
local `.env`. On the browser dashboard, paste the operator token into the
**API token** field and click **Connect**. Wait for “Connected as operator”
before uploading video, starting WebRTC, switching video sources, or changing
traffic settings. A viewer token only allows protected read requests. The
dashboard now checks the token before enabling authenticated requests and
displays the API's error instead of sending unauthenticated actions.

Copy the appropriate token from `.env` into the browser dashboard, API
Explorer, or read-only Streamlit viewer. Keep `.env` private; never paste
tokens into source control, screenshots, issue trackers, or chat. Re-run the setup script
to rotate both tokens. These are shared role tokens, not individual accounts;
there is no per-user identity, revocation, or audit trail.

For a local demonstration only, token prompts can be disabled with
`--local-demo-no-auth`. The option is rejected unless `--host` is `localhost`
or a loopback IP, and it disables authentication for local API actions. Never
use it on a network-accessible host; normal token authentication remains
enabled by default.

Uploads require an operator token, an MP4/AVI/MOV/MKV extension and video
content type, a decodable frame, and are limited to 500 MiB. Evidence and
violation endpoints send `Cache-Control: no-store`; this does not encrypt data
at rest. For any network deployment, put the service behind HTTPS, keep the
application ports private, use encrypted storage and backups, define a lawful
retention/deletion policy for plate and video data, and restrict file/database
permissions. The included app is a local prototype; these controls alone do
not make it production-ready or compliant with any jurisdiction's privacy law.

## Indian helmet dataset

The [Indian Helmet Detection dataset on Kaggle](https://www.kaggle.com/datasets/aryanvaid13/indian-helmet-detection-dataset)
is installed at `datasets/indian_helmet_detection`. It contains 942 Indian-road
images with YOLO annotations: 800 training images and 142 validation images.
Its five classes are number plates, riders, and faces with no, good, or bad
helmets. The current trial checkpoint is
`runs/indian_helmet_trial/yolo26n/weights/best.pt`; it was fine-tuned for 20
epochs on CPU and evaluated on the validation split. See
`reports/indian_helmet_validation.json` for precision, recall, F1, mAP, and
per-class metrics, and `reports/indian_helmet_video_demo.mp4` for a sampled
video preview.

The validation set is useful for a preliminary check, not an independent test:
this Kaggle export has no held-out test split. Its overall mAP@0.50 in the
current trial was about 0.55, and the rare `faceWithBadHelmet` class scored
about 0.12. Collect permitted, representative footage, annotate a separate
held-out set, and evaluate it before using the model for any operational
decision. Kaggle lists the license as unknown, while the included Roboflow
attribution file says CC BY 4.0; verify applicable terms before redistribution
or deployment.

To repeat the validation with the trained checkpoint:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_yolo `
  --images "datasets\indian_helmet_detection\valid\images" `
  --labels "datasets\indian_helmet_detection\valid\labels" `
  --model "runs\indian_helmet_trial\yolo26n\weights\best.pt" `
  --output "reports\indian_helmet_validation.json"
```

The trial checkpoint is separate; the app's default COCO model was not
replaced. Never use an unvalidated automated detection as the sole basis for
enforcement or emergency response.

## Train YOLO26n for Indian road vehicles

The [Indian Driving Dataset-Detections (YOLOv11 format) on Kaggle](https://www.kaggle.com/datasets/redzapdos123/indian-driving-dataset-detections-yolov11)
contains about 42,000 labeled images across 15 classes, including cars,
motorcycles, autorickshaws, buses, trucks, bicycles, and riders. The dataset
download is about 22 GB. It is intended for academic/non-commercial research;
review the dataset's current terms before use or redistribution.

Set up Kaggle's local OAuth login (or another supported local API credential)
without sharing tokens in chat or committing them:

```powershell
.\.venv\Scripts\python.exe -m pip install kaggle
.\.venv\Scripts\python.exe -m kaggle auth login
```

Then download and extract the dataset:

```powershell
.\.venv\Scripts\python.exe -m kaggle datasets download `
  redzapdos123/indian-driving-dataset-detections-yolov11 `
  -p datasets\indian_driving_dataset_idd `
  --unzip
```

Train from the COCO-pretrained YOLO26n checkpoint. The `batch=-1` option lets
Ultralytics select a batch size for available GPU memory; use a CUDA-enabled
PyTorch installation for `device=0`. This workspace currently has CPU-only
PyTorch even though an NVIDIA GPU is present. Install a compatible Windows
CUDA build using the [official PyTorch selector](https://pytorch.org/get-started/locally/),
then verify `torch.cuda.is_available()` returns `True`. On a CPU-only
installation, change it to `device=cpu`, but expect training on this full
dataset to take substantially longer.

```powershell
.\.venv\Scripts\python.exe -m scripts.train_yolo_vehicles `
  --epochs 100 --imgsz 640 --batch -1 --device 0 `
  --workers 4 --patience 20 --seed 42
```

The best checkpoint is saved to
`runs\indian_vehicle_training\yolo26n\weights\best.pt`. After training, evaluate
on the dataset's held-out test split (not the validation split used during
training). The dataset README says its labeled test split was created from
original training data rather than the official test data, so treat this as an
internal holdout and also evaluate on separately collected, representative
footage before making accuracy claims:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_yolo `
  --images datasets\indian_driving_dataset_idd\IDDDetectionsYOLODataset\test\images `
  --labels datasets\indian_driving_dataset_idd\IDDDetectionsYOLODataset\test\labels `
  --model runs\indian_vehicle_training\yolo26n\weights\best.pt `
  --output reports\indian_vehicle_test.json
```

This vehicle checkpoint is kept separate from the app's default COCO model.
Review the per-class metrics and representative false positives/negatives
before selecting it for application inference. The dataset's 15 classes differ
from COCO, so downstream features must be checked against those class names.

The e-challan gateway defaults to local sandbox simulation. Selecting
`--env PRODUCTION` alone does not enable external transmission: dispatch is
disabled unless an authorized provider has supplied `RTO_API_BASE` (HTTPS),
`RTO_API_TOKEN`, and a unique `RTO_HMAC_SECRET` of at least 32 characters in
the private `.env` file. No government API credentials or service connection
are included with this project. Verify the provider's integration contract and
approval before enabling any real dispatch.
Citizen SOS requests create local records only; they do not contact police,
ambulance, or emergency services. Signal-control and green-corridor actions
also affect only this prototype's in-memory simulation.

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

## Collect real-world traffic data

The included `assets/models/yolo26n.pt` is a general COCO-pretrained detector;
it is not trained on this project's road-camera footage. The local databases,
violation CSV, and evidence images are application records, not a labeled
object-detection dataset. To build project-specific evaluation data, collect
traffic videos and create human-reviewed bounding-box labels.

1. **Choose a permitted source.** Prefer videos recorded with a camera or
   dashcam you are authorized to use, especially footage from a camera angle
   similar to the one this project will process. Public sources such as
   [India Driving Dataset (IDD)](https://idd.insaan.iiit.ac.in/),
   [BDD100K](https://bdd-data.berkeley.edu/), and
   [UA-DETRAC](https://detrac-db.rit.albany.edu/) may also be useful. Before
   downloading or using any source, check its current access rules, license,
   attribution requirements, and permitted use. These datasets may provide
   images and labels rather than videos, and their labels are not guaranteed
   to match this project's format or classes.
2. **Protect privacy and preserve provenance.** Only collect or retain footage
   when you have permission and a lawful basis. Road video may contain faces
   and number plates; restrict access, avoid publishing raw footage, and follow
   applicable privacy rules. Keep a note of each source, capture date, camera
   conditions, and any usage restrictions. Do not treat simulated registry
   entries or violation records as verified ground truth.
3. **Copy video files into the project.** Put authorized local videos in
   `uploads\videos`, then check their names:

   ```powershell
   Get-ChildItem .\uploads\videos -File
   ```

   The project's preparation tool accepts video file paths; it does not
   download public datasets or convert arbitrary dataset annotation formats.
   The video must be readable by OpenCV. Use distinct video filenames within
   each split because sampled image names are derived from the video filename.
4. **Separate source videos into splits before sampling.** Use different
   recordings for training, validation, and test. Do not put frames from the
   same recording in multiple splits; nearby frames are too similar and can
   make the test score misleading. For a first trial, sample a manageable
   number of frames from several different recordings, then inspect the images
   and increase coverage for underrepresented conditions such as night, rain,
   distant vehicles, and crowded roads.
5. **Sample frames using the project tool.** Replace the example paths with
   your actual filenames. Repeat each option to add more videos to that split:

   ```powershell
   .\.venv\Scripts\python.exe -m scripts.prepare_yolo_dataset `
     --train-video "uploads\videos\train_day.mp4" `
     --train-video "uploads\videos\train_night.mp4" `
     --val-video "uploads\videos\validation.mp4" `
     --test-video "uploads\videos\heldout_test.mp4" `
     --frames-per-video 100
   ```

   The tool samples approximately evenly spaced frames into
   `datasets\traffic\images\{train,val,test}`. It requires at least one
   `--test-video`. Choose different held-out test videos that represent the
   intended deployment conditions; do not use test footage to tune the model.
6. **Review and annotate all images.** Run the bundled annotator once for
   each split. Draw a box for every visible object from the supported classes:
   person, bicycle, car, motorcycle, bus, and truck. Save empty label files
   for reviewed images that contain none of those objects. The classes are
   COCO IDs `0, 1, 2, 3, 5, 7`, which match the corresponding class IDs in the
   included COCO checkpoint:

   ```powershell
   .\.venv\Scripts\python.exe -m scripts.annotate_yolo `
     --images datasets\traffic\images\train `
     --labels datasets\traffic\labels\train
   .\.venv\Scripts\python.exe -m scripts.annotate_yolo `
     --images datasets\traffic\images\val `
     --labels datasets\traffic\labels\val
   .\.venv\Scripts\python.exe -m scripts.annotate_yolo `
     --images datasets\traffic\images\test `
     --labels datasets\traffic\labels\test
   ```

   Check annotations for missed objects, incorrect classes, and loose or
   inconsistent boxes. Do not use model predictions as ground truth. If using
   a public dataset's existing annotations, first verify its class meanings,
   box format, and class IDs and convert them only when they genuinely
   correspond to these project classes.

**Important training limitation:** This workflow prepares and labels data for
evaluation of the included COCO detector. The repository does not currently
provide an end-to-end training command. Do not create a six-class training
configuration and feed it the annotator's labels unchanged: those labels use
COCO IDs `0, 1, 2, 3, 5, 7`, not consecutive IDs `0` through `5`. Custom
training requires a consistent class-ID mapping across labels, dataset
configuration, evaluation, and application inference.

## Evaluate and compare YOLO models

Once every test image has a manually reviewed label file, run evaluation only
on the held-out test split. The labels use standard YOLO detection format:
`class_id x_center y_center width height`, normalized to `[0, 1]`. Class IDs
are the corresponding COCO IDs used by the included COCO-pretrained detector.
Do not use model-generated predictions as ground truth. Evaluation requires a
label file for every image and at least one labeled object in the evaluation
set; the evaluator fails on missing labels rather than silently treating them
as negative images.

```text
datasets\traffic\images\val\highway\frame001.jpg
datasets\traffic\labels\val\highway\frame001.txt
```

After all evaluation images are reviewed and labeled, evaluate only the held-out
test split:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_yolo `
  --images datasets\traffic\images\test `
  --labels datasets\traffic\labels\test `
  --model assets\models\yolo26n.pt `
  --output reports\yolo-evaluation.json
```

The JSON report includes precision, recall, F1 at confidence `0.25`, mAP@0.5,
mAP@0.5:0.95, per-class AP, false positives/negatives, model-load time, and
mean/median/p95 inference latency. The evaluator uses a `0.001` minimum
inference confidence by default for the mAP confidence curve; tune it with
`--min-confidence` if needed, and use `--metric-confidence` to change the
precision/recall operating point. Compare models on the exact same test images
and labels. Record the test image/object counts, video sources, model checkpoint,
settings, hardware, and known limitations when presenting the metrics. The
repository does not include a pre-labeled traffic benchmark, so do not claim
evaluation results until the sampled frames have been independently reviewed
and labeled.

## Run the applications

Run the main FastAPI traffic safety system:

```powershell
.\.venv\Scripts\python.exe -m src.main
```

It prints the web dashboard, API explorer, and REST API URLs in the terminal.
Use `Ctrl+C` in that terminal to stop it.

To run without token prompts on this computer only:

```powershell
.\.venv\Scripts\python.exe -m src.main --local-demo-no-auth
```

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
├── scripts/                 Dataset preparation, annotation, evaluation, and DB utilities
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
