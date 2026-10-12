# Traffic Safety System Architecture

## Product scope

The product is a traffic-safety command center for processing a live camera,
video file, RTSP stream, or simulator feed; presenting detection and telemetry
to operators; and providing APIs for traffic operations, citizen reports,
vehicle lookup, and e-challan records. It also includes a separate,
read-only dashboard for reviewing persisted records and a workflow for
preparing and training the vehicle detector.

This document describes the implementation in this repository. The runtime is
a Python application, not a collection of independently deployed services.

## System context

```mermaid
flowchart LR
    Camera["Camera / RTSP / video file"] --> Runtime["Traffic Safety API<br/>FastAPI + YOLO pipeline"]
    Simulator["Built-in traffic simulator"] --> Runtime
    Operator["Traffic operator"] --> WebUI["Operations dashboard<br/>FastAPI-served web UI"]
    WebUI -->|REST controls and queries| Runtime
    WebUI <-->|WebSocket telemetry| Runtime
    WebUI <-->|WebRTC offer / video| Runtime
    Explorer["API Explorer<br/>FastAPI-served web UI"] -->|REST calls| Runtime
    Citizen["Citizen / external client"] -->|REST| Runtime
    Runtime --> Storage["SQLite files or PostgreSQL<br/>registry, reports, telemetry, ledger"]
    Runtime --> Logs["Violation CSV and evidence images"]
    Runtime --> RTO["RTO / e-challan gateway<br/>SANDBOX by default"]
    RecordsUI["Read-only records dashboard<br/>Streamlit"] --> Storage
    RecordsUI --> Logs
    Dataset["Licensed images + YOLO labels"] --> Prepare["Dataset preparation"]
    Prepare --> DatasetSplits["Train / validation / test dataset"]
    DatasetSplits --> Train["YOLO training and evaluation"]
    Train --> Weights["Trained model weights"]
    Weights -. configured at runtime .-> Runtime
```

## Project flowcharts

### Whole-project structure and data flow

This view connects the product's user-facing apps, runtime, persisted data,
and offline detector development workflow.

```mermaid
flowchart TB
    subgraph Inputs["Inputs and users"]
        Camera["Camera / RTSP"]
        Video["Video file"]
        Simulator["Traffic simulator"]
        Operator["Traffic operator"]
        Citizen["Citizen / API client"]
        Images["Authorized road images + YOLO labels"]
    end

    subgraph Apps["Application entry points"]
        Main["src/main.py<br/>FastAPI runtime"]
        OpsUI["Operations dashboard<br/>FastAPI-served UI"]
        ApiUI["API Explorer<br/>FastAPI-served UI"]
        RecordsUI["dashboard/app.py<br/>Streamlit read-only viewer"]
    end

    subgraph Processing["In-process traffic processing"]
        Ingest["Frame grabber"]
        Enhance["Weather analysis<br/>image enhancement + ROI"]
        Detect["YOLO tracking"]
        Analyze["Vehicle speed, trajectory,<br/>risk and incident analysis"]
        Operations["Signals, corridor preemption,<br/>hotlist and safety modules"]
        Violation["Violation / challan handling"]
    end

    subgraph Persistence["Persisted outputs"]
        DB["SQLite files or PostgreSQL"]
        CSV["Violation CSV"]
        Evidence["Evidence snapshots"]
        Uploads["Uploaded video / citizen media"]
    end

    subgraph Training["Offline detector lifecycle"]
        Prepare["scripts/prepare_dataset.py"]
        Splits["Train / validation / test"]
        Train["scripts/train_detector.py"]
        Model["assets/models/<br/>selected YOLO weights"]
    end

    RTO["Sandbox or configured RTO gateway"]
    Browser["Operator browser"]

    Camera --> Ingest
    Video --> Ingest
    Simulator --> Ingest
    Main --> Ingest
    Ingest --> Enhance --> Detect --> Analyze
    Analyze --> Operations
    Analyze --> Violation
    Operations --> Violation
    Violation --> DB
    Violation --> CSV
    Violation --> Evidence
    Violation --> RTO
    Citizen --> Main
    Main --> DB
    Main --> Uploads
    Operator --> Browser
    Browser --> OpsUI
    Browser --> ApiUI
    OpsUI <-->|REST, WebSocket, WebRTC| Main
    ApiUI -->|REST| Main
    RecordsUI -->|read-only| DB
    RecordsUI -->|read-only| CSV
    Images --> Prepare --> Splits --> Train --> Model
    Model -. configured for inference .-> Detect
```

### Live traffic and violation flow

```mermaid
flowchart TD
    Start["Select camera, RTSP, video, or simulator"] --> Frame["Read next frame"]
    Frame --> Weather["Estimate visibility / weather"]
    Weather --> Prepare["Enhance frame and apply ROI mask"]
    Prepare --> Track["YOLO track objects"]
    Track --> Classes{"Classify tracked objects"}

    Classes -->|Vehicle| Motion["Update track history and lane"]
    Motion --> Radar{"Crosses both calibrated radar lines?"}
    Radar -->|Yes| Speed["Estimate speed from distance and elapsed time"]
    Radar -->|No| Risk
    Speed --> Limit{"Speed over configured limit?"}
    Limit -->|Yes| Violation["Build violation / challan event"]
    Limit -->|No| Risk["Evaluate predictive risk and incidents"]
    Risk --> Alert{"Risk or incident alert?"}
    Alert -->|Yes| Violation
    Alert -->|No| Telemetry

    Classes -->|Person or animal| Crossing["Check crossing danger zone"]
    Crossing --> Signals["Update adaptive signal timing"]
    Signals --> Telemetry["Refresh telemetry and annotated frame"]
    Violation --> Profile["Resolve vehicle profile, HSRP and hotlist"]
    Profile --> Fine["Select violation details and fine"]
    Fine --> Outputs["Save evidence + CSV + telemetry"]
    Fine --> Ledger["Sign challan and append ledger block"]
    Ledger --> Queue["Queue asynchronous RTO dispatch"]
    Queue --> Mode{"Gateway mode"}
    Mode -->|SANDBOX| Simulate["Simulate dispatch"]
    Mode -->|PRODUCTION| External["POST to configured RTO endpoint"]
    Outputs --> Telemetry
    Simulate --> Telemetry
    External --> Telemetry

    Telemetry --> Clients["Dashboard: WebSocket telemetry / WebRTC video"]
    Clients --> Frame
```

The radar-line speed estimate depends on configured distance and timestamps.
The diagram describes the implemented processing path; model detections alone
do not establish every legal violation.

### Development and deployment flow

```mermaid
flowchart LR
    Source["src/main.py + src/config.py"] --> Runtime["Run FastAPI traffic system"]
    Runtime --> APIs["REST / OpenAPI"]
    Runtime --> Live["Operations UI + live video / telemetry"]
    Runtime --> Records["SQLite or PostgreSQL records"]
    Records --> Viewer["Run Streamlit records viewer separately"]
    Tests["tests/"] --> Verify["Run unittest discovery"]

    Labels["Source images + labels"] --> Prepare["scripts/prepare_dataset.py"]
    Prepare --> Dataset["Validated reproducible splits + dataset.yaml"]
    Dataset --> Train["scripts/train_detector.py"]
    Train --> Weights["Trained best.pt + evaluation metrics"]
    Weights --> Runtime
```

## Runtime components

### Traffic processing

`src/main.py` hosts the runtime pipeline and the HTTP/WebSocket endpoints.
At startup, the FastAPI lifespan:

1. Creates a `DynamicTrafficGrabber` for the selected source. If a usable
   camera/stream source is unavailable, the pipeline can use its simulator.
2. Creates an `APPoliceRadarSafetyEngine` with the configured YOLO model.
   The engine tracks objects and calculates detection and traffic telemetry.
3. Starts the frame grabber, detection engine, acoustic analysis worker, and
   asynchronous RTO dispatch worker.
4. Publishes telemetry to connected WebSocket clients and exposes the latest
   video through WebRTC.
5. Stops workers and closes peer connections during application shutdown.

The detector pipeline combines the YOLO tracker with ROI/preprocessing,
trajectory and collision analysis, plate/vehicle lookup, and violation
processing. Other in-process engines provide:

- Predictive risk and incident alerts, including time-to-collision,
  stalled-vehicle, wrong-way, and overlapping-box checks.
- Adaptive signals and emergency green-corridor preemption.
- HSRP-format checks, vehicle registry resolution, and CCTNS hotlist lookup.
- Citizen hazard reports and SOS events.
- Weather and fleet-emissions analysis.
- Optimization, registry-search, and hazard-classification endpoints grouped
  under the QAOA, Grover, and QSVM module names.

These engines are instantiated inside the main application process. The
quantum-named modules are application modules in this repository; the runtime
does not connect to a separately deployed quantum-computing service.

### HTTP and user interfaces

The process runs three FastAPI applications on separate configurable ports:

| Application | Purpose | Default port |
| --- | --- | ---: |
| Main API | REST API, OpenAPI/Swagger, WebSocket telemetry, and WebRTC signaling | 8000 |
| Operations dashboard | Browser UI for stream controls, live monitoring, and safety operations | 8080 |
| API Explorer | Browser UI generated from the main API's OpenAPI schema | 8081 |

The dashboard and API Explorer are auxiliary FastAPI apps started in server
threads by `src.main`; they call the main API over HTTP. Browser-origin access
is configured with `DASHBOARD_ORIGINS`. The separate `dashboard/app.py`
application is a Streamlit records viewer, started independently, and reads
records without writing them.

The main API includes health and overview, detections, video upload/source
switching, snapshot and violation export, vehicle lookup, traffic-signal
controls, blockchain ledger, citizen report/SOS, HSRP checks, green-corridor
controls, weather/carbon reports, and quantum-named module operations.

## Data and persistence

The application uses a small database adapter to support either SQLite or
PostgreSQL:

- If `DATABASE_URL` is set, runtime database connections use PostgreSQL through
  `psycopg`.
- Otherwise, runtime data is stored in project-relative SQLite files under
  `SQLITE_DATABASE_DIR` (default: `data/`).
- The runtime initializes its application tables when its engines are created.
  PostgreSQL and existing SQLite files are separate data stores; selecting a
  PostgreSQL URL does not migrate/import the SQLite records.

| Data | Default SQLite file | Main owner |
| --- | --- | --- |
| Blockchain-style e-challan blocks | `data/challan_blockchain.db` | `ChallanBlockchainLedger` |
| Citizen safety reports | `data/citizen_safety.db` | `CitizenSafetyPortalEngine` |
| Vehicle registry | `data/traffic_registry.db` | `APVehicleRegistryEngine` |
| Highway telemetry | `data/traffic_timeseries.db` | `TimeSeriesMetricsLogger` |
| Violation export/log | `data/traffic_violations_log.csv` | Traffic processing pipeline |
| Violation snapshots | `evidence/violations/` | Traffic processing pipeline |
| Uploaded videos and citizen media | `uploads/videos/`, `uploads/citizen_uploads/` | Upload/API flows |

The ledger chains records with SHA-256 hashes; it is an application-managed
ledger stored in the configured database, not a connection to a public
blockchain network.

The Streamlit viewer reads the four database tables and violation CSV. It
limits displayed records and uses short-lived caching; it is not part of the
FastAPI lifespan.

## e-Challan dispatch flow

```mermaid
sequenceDiagram
    participant V as Detector / violation pipeline
    participant G as RTO integration gateway
    participant L as Challan ledger
    participant Q as Outbound queue
    participant E as External RTO endpoint

    V->>G: Submit challan payload
    G->>G: Sign payload with HMAC-SHA256
    G->>L: Mint and persist ledger block
    L-->>G: Block index and hash
    G->>Q: Queue signed challan
    Q->>E: Dispatch asynchronously in PRODUCTION
    E-->>Q: HTTP result
```

The default gateway mode is `SANDBOX`; it simulates successful dispatch rather
than contacting the external authority. `PRODUCTION` dispatch uses the
configured integration endpoint. Real enforcement or external dispatch should
be enabled only after the endpoint, credentials, operational procedures, and
decision rules have been validated.

## Detector data and model lifecycle

The detector lifecycle is separate from live application startup:

1. Collect/authorize images and create YOLO-format labels.
2. Run `scripts/prepare_dataset.py` to validate annotations and create
   reproducible train/validation/test splits and a dataset YAML file.
3. Run `scripts/train_detector.py` to fine-tune YOLO and evaluate the selected
   validation or test split.
4. Point the runtime `--model` option at the resulting weights.

The trained model provides object detections/tracks. It does not, by itself,
establish that a statutory violation occurred. The project documentation
notes that speed and signal-violation decisions need calibrated tracking and
signal/stop-line logic, and helmet/signal-jumping challan decisions are not
currently driven by trained model detections.

## Configuration and local operation

- `.env` is loaded from the project root; `.env.example` documents local
  settings. Do not commit database credentials.
- `--source` accepts a camera index, RTSP URL, or video path.
- `--model`, `--host`, `--port`, `--dashboard-port`, and
  `--api-interface-port` configure the runtime.
- `--env SANDBOX|PRODUCTION` selects the RTO gateway behavior.
- `DASHBOARD_API_URL`, `DASHBOARD_UI_URL`,
  `DASHBOARD_API_INTERFACE_URL`, and `DASHBOARD_ORIGINS` configure browser/API
  connectivity when components are accessed across hosts.
- Start the Streamlit records viewer separately with
  `streamlit run dashboard/app.py`.

## Repository map

| Path | Responsibility |
| --- | --- |
| `src/main.py` | FastAPI apps, traffic ingestion/detection, in-process engines, storage access, and runtime APIs |
| `src/config.py` | Minimal project-root `.env` loading |
| `dashboard/app.py` | Read-only Streamlit database and violation-log viewer |
| `scripts/prepare_dataset.py` | Annotation validation and dataset splitting |
| `scripts/train_detector.py` | YOLO fine-tuning and evaluation |
| `assets/models/` | Runtime detector weights |
| `data/` | Local database files and violation CSV |
| `uploads/`, `evidence/` | Uploaded media and generated evidence |
| `tests/` | Automated regression tests |
