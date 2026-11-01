import json

from fastapi.responses import HTMLResponse

if __package__ == "src.web":
    from src.config import DASHBOARD_API_INTERFACE_URL, DASHBOARD_API_URL, DASHBOARD_UI_URL
else:
    from config import DASHBOARD_API_INTERFACE_URL, DASHBOARD_API_URL, DASHBOARD_UI_URL


def build_api_interface_page(openapi_schema, api_url, dashboard_url):
    schema_json = json.dumps(openapi_schema).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    api_base_json = json.dumps(api_url)
    dashboard_base_json = json.dumps(dashboard_url)
    page = r"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Traffic Safety API Explorer</title>
    <style>
        :root { color-scheme: light; --bg:#f4f7fb; --surface:#fff; --line:#e1e7ef; --text:#172033; --muted:#667085; --blue:#2459d3; --green:#137a52; --red:#b42318; }
        * { box-sizing:border-box }
        body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif }
        header { background:#fff; border-bottom:1px solid var(--line); padding:22px max(20px,calc((100vw - 1180px)/2)); display:flex; justify-content:space-between; align-items:center; gap:16px }
        h1 { margin:0; font-size:1.45rem } h2 { margin:0 0 8px } h3 { margin:0; font-size:1rem }
        main { max-width:1180px; margin:auto; padding:24px 20px 56px }
        p { margin:5px 0; color:var(--muted) }
        a { color:var(--blue); text-decoration:none } a:hover { text-decoration:underline }
        .toolbar,.operation,.intro { background:var(--surface); border:1px solid var(--line); border-radius:12px; box-shadow:0 3px 12px #182b4810 }
        .toolbar { padding:16px; display:grid; grid-template-columns:2fr 2fr 1fr; gap:12px; margin:18px 0 }
        label { display:block; font-size:.8rem; font-weight:650; color:var(--muted); margin-bottom:5px }
        input,textarea,button { font:inherit }
        input,textarea { width:100%; border:1px solid #cdd5df; border-radius:7px; padding:9px 10px; background:#fff; color:var(--text) }
        textarea { min-height:112px; resize:vertical; font:13px/1.5 ui-monospace,Consolas,monospace }
        button,.button { display:inline-flex; align-items:center; justify-content:center; border:1px solid #cbd5e1; border-radius:7px; padding:8px 12px; background:#fff; color:var(--text); cursor:pointer; font-weight:650 }
        button:hover,.button:hover { border-color:var(--blue); text-decoration:none }
        button.primary { background:var(--blue); border-color:var(--blue); color:#fff }
        .intro { padding:17px; margin-bottom:18px }
        .summary { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:10px; color:var(--muted); font-size:.88rem }
        #operations { display:grid; gap:10px }
        .operation { overflow:hidden }
        .operation summary { list-style:none; cursor:pointer; padding:14px 16px; display:flex; gap:10px; align-items:center }
        .operation summary::-webkit-details-marker { display:none }
        .method { min-width:76px; text-align:center; border-radius:6px; padding:4px 7px; font-size:.76rem; font-weight:800; letter-spacing:.03em; background:#edf2f7; color:#344054 }
        .method.get { background:#e8f1ff; color:#175cd3 } .method.post { background:#e8f8ef; color:#137a52 }
        .method.put,.method.patch { background:#fff5df; color:#a15c00 } .method.delete { background:#ffebe9; color:#b42318 }
        .method.websocket { background:#f1eafe; color:#6941c6 }
        .path { font:600 .91rem ui-monospace,Consolas,monospace; overflow-wrap:anywhere }
        .op-summary { color:var(--muted); margin-left:auto; font-size:.86rem; text-align:right }
        .details { border-top:1px solid var(--line); padding:16px }
        .description { margin:0 0 13px }
        .params { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; margin-bottom:12px }
        .param .location { font-size:.72rem; color:var(--muted); text-transform:uppercase }
        .body-editor { margin:12px 0 }
        .actions { display:flex; flex-wrap:wrap; gap:8px; align-items:center }
        .response { margin-top:12px; padding:12px; border-radius:8px; background:#f8fafc; border:1px solid var(--line); overflow:auto }
        .response strong.ok { color:var(--green) } .response strong.bad { color:var(--red) }
        pre { margin:8px 0 0; white-space:pre-wrap; overflow-wrap:anywhere; font:13px/1.5 ui-monospace,Consolas,monospace }
        .count { color:var(--muted); font-size:.88rem }
        @media(max-width:700px) { header { align-items:flex-start; flex-direction:column } .toolbar { grid-template-columns:1fr } .op-summary { display:none } }
    </style>
</head>
<body>
<header><div><h1>Traffic Safety API Explorer</h1><p>Browse and call every documented API operation.</p></div><a class="button" href="__DASHBOARD_URL__">Back to dashboard</a></header>
<main>
    <section class="intro">
        <h2>API catalogue</h2>
        <p>The endpoint list is generated from this service's live OpenAPI schema, so newly registered routes appear automatically. Requests run against the configured API server.</p>
        <div class="summary"><span id="count" class="count"></span><a id="docsLink" href="#">Open interactive Swagger docs</a><span>·</span><a id="schemaLink" href="#">Open raw OpenAPI schema</a></div>
    </section>
    <section class="toolbar" aria-label="API request settings">
        <div><label for="apiBase">API base URL</label><input id="apiBase" type="url" spellcheck="false"></div>
        <div><label for="token">Bearer token (optional)</label><input id="token" type="password" autocomplete="off" placeholder="Paste an access token for protected routes"></div>
        <div><label for="search">Filter endpoints</label><input id="search" type="search" placeholder="Path, method, or name"></div>
    </section>
    <div id="operations"></div>
</main>
<script>
const apiSchema = __OPENAPI_SCHEMA__;
const configuredApiBase = __API_BASE__;
const configuredDashboardUrl = __DASHBOARD_URL__;
const baseInput = document.getElementById('apiBase');
baseInput.value = configuredApiBase || window.location.origin;
document.querySelector('header .button').href = configuredDashboardUrl || window.location.origin;
document.getElementById('docsLink').href = new URL('/docs', baseInput.value).href;
document.getElementById('schemaLink').href = new URL('/openapi.json', baseInput.value).href;
baseInput.addEventListener('change', () => {
    document.getElementById('docsLink').href = new URL('/docs', baseInput.value).href;
    document.getElementById('schemaLink').href = new URL('/openapi.json', baseInput.value).href;
});
const operationsRoot = document.getElementById('operations');
const httpMethods = new Set(['get','post','put','patch','delete','options','head']);
const methods = [];
for (const [path, pathItem] of Object.entries(apiSchema.paths || {})) {
    for (const method of httpMethods) {
        const operation = pathItem[method];
        if (operation) methods.push({path, method, operation, parameters:[...(pathItem.parameters || []), ...(operation.parameters || [])]});
    }
}
methods.sort((a,b) => a.path.localeCompare(b.path) || a.method.localeCompare(b.method));
document.getElementById('count').textContent = `${methods.length + 1} operations, including the telemetry WebSocket`;

function text(tag, value, className) {
    const node = document.createElement(tag);
    node.textContent = value || '';
    if (className) node.className = className;
    return node;
}
function resolveSchema(schema) {
    if (!schema) return {};
    if (schema.$ref) {
        const parts = schema.$ref.slice(2).split('/');
        const target = parts.reduce(
            (value, key) => value && value[key.replace(/~1/g, '/').replace(/~0/g, '~')],
            apiSchema
        );
        return resolveSchema(target);
    }
    if (schema.allOf) return Object.assign({}, ...schema.allOf.map(resolveSchema));
    return schema;
}
function exampleValue(schema, depth=0) {
    schema = resolveSchema(schema);
    if (schema.example !== undefined) return schema.example;
    if (schema.default !== undefined) return schema.default;
    if (schema.enum && schema.enum.length) return schema.enum[0];
    if (depth > 4) return null;
    if (schema.type === 'object' || schema.properties) {
        const value = {};
        for (const [key, property] of Object.entries(schema.properties || {})) {
            if (!schema.required || schema.required.includes(key)) value[key] = exampleValue(property, depth+1);
        }
        return value;
    }
    if (schema.type === 'array') return [];
    if (schema.type === 'integer' || schema.type === 'number') return schema.minimum || 0;
    if (schema.type === 'boolean') return false;
    if (schema.type === 'string') return '';
    return {};
}
function addParamField(container, parameter) {
    const wrap = document.createElement('div');
    wrap.className = 'param';
    const label = text('label', parameter.name + (parameter.required ? ' *' : ''));
    label.htmlFor = `param-${parameter.in}-${parameter.name}`;
    const input = document.createElement('input');
    input.id = label.htmlFor;
    input.dataset.name = parameter.name;
    input.dataset.location = parameter.in;
    input.required = Boolean(parameter.required);
    const schema = resolveSchema(parameter.schema);
    input.placeholder = schema.description || (schema.enum ? schema.enum.join(', ') : '');
    if (schema.default !== undefined) input.value = schema.default;
    else if (schema.enum && schema.enum.length) input.value = schema.enum[0];
    const location = text('div', parameter.in, 'location');
    wrap.append(label, input, location);
    container.append(wrap);
}
function makeCard(item) {
    const {path, method, operation, parameters} = item;
    const card = document.createElement('details');
    card.className = 'operation';
    const summary = document.createElement('summary');
    summary.append(text('span', method.toUpperCase(), `method ${method}`), text('span', path, 'path'),
        text('span', operation.summary || (operation.tags || []).join(', ') || 'API operation', 'op-summary'));
    card.append(summary);
    const details = document.createElement('div');
    details.className = 'details';
    const description = text('p', operation.description || '', 'description');
    if (description.textContent) details.append(description);
    const fields = document.createElement('div');
    fields.className = 'params';
    parameters.forEach(parameter => addParamField(fields, parameter));
    if (fields.childElementCount) details.append(fields);
    const requestBody = operation.requestBody;
    const content = requestBody && requestBody.content || {};
    const mediaType = Object.keys(content)[0] || '';
    const bodyEditor = document.createElement('div');
    bodyEditor.className = 'body-editor';
    let bodyInput = null;
    let formFields = [];
    if (mediaType === 'multipart/form-data' || mediaType === 'application/x-www-form-urlencoded') {
        const schema = resolveSchema(content[mediaType].schema);
        for (const [name, propertySchema] of Object.entries(schema.properties || {})) {
            const wrap = document.createElement('div');
            wrap.className = 'param';
            const label = text('label', name + ((schema.required || []).includes(name) ? ' *' : ''));
            const input = document.createElement('input');
            input.dataset.name = name;
            input.dataset.file = propertySchema.format === 'binary' ? 'true' : 'false';
            input.type = propertySchema.format === 'binary' ? 'file' : 'text';
            input.required = (schema.required || []).includes(name);
            if (propertySchema.default !== undefined) input.value = propertySchema.default;
            wrap.append(label, input);
            formFields.push(input);
            bodyEditor.append(wrap);
        }
    } else if (requestBody || ['post','put','patch'].includes(method)) {
        bodyEditor.append(text('label', `Request body${mediaType ? ` · ${mediaType}` : ' · application/json'}`));
        bodyInput = document.createElement('textarea');
        bodyInput.spellcheck = false;
        bodyInput.value = JSON.stringify(requestBody ? exampleValue(content[mediaType].schema) : {}, null, 2);
        bodyEditor.append(bodyInput);
    }
    if (bodyEditor.childElementCount) details.append(bodyEditor);
    const actions = document.createElement('div');
    actions.className = 'actions';
    const send = text('button', 'Send request', 'primary');
    send.type = 'button';
    actions.append(send);
    const response = document.createElement('div');
    response.className = 'response';
    response.hidden = true;
    details.append(actions, response);
    card.append(details);
    send.addEventListener('click', async event => {
        event.preventDefault();
        try {
            let requestPath = path.replace(/\{([^}]+)\}/g, (_, name) => {
                const input = [...fields.querySelectorAll('input')].find(field => field.dataset.location === 'path' && field.dataset.name === name);
                if (!input || !input.value) throw new Error(`Enter the required path parameter: ${name}`);
                return encodeURIComponent(input.value);
            });
            const url = new URL(requestPath.replace(/^\/+/, ''), baseInput.value.replace(/\/?$/, '/'));
            for (const input of fields.querySelectorAll('input')) {
                if (!input.value) {
                    if (input.required) throw new Error(`Enter the required ${input.dataset.location} parameter: ${input.dataset.name}`);
                    continue;
                }
                if (input.dataset.location === 'query') url.searchParams.set(input.dataset.name, input.value);
            }
            const headers = {};
            for (const input of fields.querySelectorAll('input')) {
                if (input.dataset.location === 'header' && input.value) headers[input.dataset.name] = input.value;
            }
            const token = document.getElementById('token').value.trim();
            if (token) headers.Authorization = `Bearer ${token}`;
            const init = {method:method.toUpperCase(), headers};
            if (formFields.length) {
                if (mediaType === 'application/x-www-form-urlencoded') {
                    const form = new URLSearchParams();
                    for (const input of formFields) {
                        if (input.value) form.append(input.dataset.name, input.value);
                    }
                    init.headers['Content-Type'] = mediaType;
                    init.body = form;
                } else {
                    const form = new FormData();
                    for (const input of formFields) {
                        if (input.dataset.file === 'true') {
                            if (input.files[0]) form.append(input.dataset.name, input.files[0]);
                        } else if (input.value) form.append(input.dataset.name, input.value);
                    }
                    init.body = form;
                }
            } else if (bodyInput) {
                const value = bodyInput.value.trim();
                if (value) {
                    init.headers['Content-Type'] = mediaType || 'application/json';
                    init.body = value;
                }
            }
            response.hidden = false;
            response.replaceChildren(text('strong', 'Sending…'));
            const result = await fetch(url, init);
            const responseText = await result.text();
            let formatted = responseText;
            try { formatted = JSON.stringify(JSON.parse(responseText), null, 2); } catch {}
            response.replaceChildren(text('strong', `${result.status} ${result.statusText}`, result.ok ? 'ok' : 'bad'), text('pre', formatted));
        } catch (error) {
            response.hidden = false;
            response.replaceChildren(text('strong', 'Request could not be sent', 'bad'), text('pre', error.message));
        }
    });
    return card;
}
methods.forEach(item => operationsRoot.append(makeCard(item)));
const websocketItem = document.createElement('details');
websocketItem.className = 'operation';
const websocketSummary = document.createElement('summary');
websocketSummary.append(text('span', 'WEBSOCKET', 'method websocket'), text('span', '/ws/telemetry', 'path'), text('span', 'Live telemetry settings', 'op-summary'));
websocketItem.append(websocketSummary);
const websocketDetails = document.createElement('div');
websocketDetails.className = 'details';
websocketDetails.append(text('p', 'Connect to the telemetry socket and send a JSON settings object. The socket accepts speed_limit, distance_m, conf, line1_y, and line2_y.'));
const websocketBody = document.createElement('textarea');
websocketBody.value = JSON.stringify({speed_limit:60,distance_m:20,conf:0.3,line1_y:240,line2_y:520}, null, 2);
const websocketButton = text('button', 'Connect and send settings', 'primary');
websocketButton.type = 'button';
const websocketResponse = document.createElement('div');
websocketResponse.className = 'response';
websocketResponse.hidden = true;
websocketDetails.append(websocketBody, websocketButton, websocketResponse);
websocketItem.append(websocketDetails);
operationsRoot.append(websocketItem);
websocketButton.addEventListener('click', event => {
    event.preventDefault();
    try {
        const settings = JSON.parse(websocketBody.value);
        const apiUrl = new URL(baseInput.value);
        apiUrl.protocol = apiUrl.protocol === 'https:' ? 'wss:' : 'ws:';
        apiUrl.pathname = '/ws/telemetry';
        apiUrl.search = '';
        const socket = new WebSocket(apiUrl);
        websocketResponse.hidden = false;
        websocketResponse.replaceChildren(text('strong', 'Connecting…'));
        socket.addEventListener('open', () => {
            socket.send(JSON.stringify(settings));
            websocketResponse.replaceChildren(text('strong', 'Connected; settings sent.', 'ok'));
            window.setTimeout(() => socket.close(), 700);
        }, {once:true});
        socket.addEventListener('error', () => websocketResponse.replaceChildren(text('strong', 'WebSocket connection failed.', 'bad')), {once:true});
    } catch (error) {
        websocketResponse.hidden = false;
        websocketResponse.replaceChildren(text('strong', 'Request could not be sent', 'bad'), text('pre', error.message));
    }
});
document.getElementById('search').addEventListener('input', event => {
    const query = event.target.value.trim().toLowerCase();
    for (const card of operationsRoot.children) card.hidden = !card.textContent.toLowerCase().includes(query);
});
</script>
</body>
</html>""".replace("__OPENAPI_SCHEMA__", schema_json).replace("__API_BASE__", api_base_json).replace(
        "__DASHBOARD_URL__", dashboard_base_json
    )
    return HTMLResponse(page)


def build_dashboard_page(api_url, api_interface_url):
    dashboard_html = """<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>Real-Time Multi-Hazard Traffic Surveillance and Automated e-Challan System using Edge YOLO and WebRTC</title>  
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">  
    <style>  
        :root {  
            --bg: #05070d;  
            --surface: #0e1320;  
            --surface-border: #1a2236;  
            --primary: #2563eb;  
            --accent-cyan: #06b6d4;  
            --accent-green: #10b981;  
            --accent-red: #ef4444;  
            --accent-gold: #f59e0b;  
            --accent-purple: #8b5cf6;  
            --text: #f8fafc;  
            --text-muted: #94a3b8;  
        }  
        * { box-sizing: border-box; }  
        body {  
            font-family: 'Space Grotesk', sans-serif;  
            background: var(--bg);  
            color: var(--text);  
            margin: 0;  
            padding: 1.2rem;  
            display: flex;  
            flex-direction: column;  
            align-items: center;  
        }  
        .header {  
            width: 100%;  
            max-width: 1540px;  
            display: flex;  
            justify-content: space-between;  
            align-items: center;  
            margin-bottom: 0.8rem;  
            padding-bottom: 0.8rem;  
            border-bottom: 1px solid var(--surface-border);  
        }  
        .system-overview {
            width: 100%;
            max-width: 1540px;
            display: grid;
            grid-template-columns: repeat(4, minmax(180px, 1fr));
            gap: 0.8rem;
            margin: 0.8rem 0 1rem;
        }
        .overview-card {
            background: linear-gradient(180deg, rgba(15,23,42,0.92), rgba(15,23,42,0.66));
            border: 1px solid var(--surface-border);
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
            min-height: 92px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .overview-label {
            font-size: 0.7rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-muted);
        }
        .overview-value {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text);
            font-family: 'JetBrains Mono', monospace;
        }
        .overview-delta {
            font-size: 0.72rem;
            color: var(--accent-green);
        }
        .header-title h1 { margin: 0; font-size: 1.35rem; font-weight: 700; color: #fff; }  
        .header-title p { margin: 0.2rem 0 0; color: var(--text-muted); font-size: 0.85rem; }  
        .header-badges { display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; }  
        .api-explorer-link {
            display: inline-flex;
            align-items: center;
            padding: 0.4rem 0.8rem;
            border: 1px solid rgba(37, 99, 235, 0.35);
            border-radius: 9999px;
            background: rgba(37, 99, 235, 0.12);
            color: #93c5fd;
            font-size: 0.8rem;
            font-weight: 700;
            text-decoration: none;
        }
        .api-explorer-link:hover { background: rgba(37, 99, 235, 0.22); }
        .badge-radar {  
            display: flex;  
            align-items: center;  
            gap: 0.5rem;  
            background: rgba(16, 185, 129, 0.1);  
            color: var(--accent-green);  
            padding: 0.4rem 0.8rem;  
            border-radius: 9999px;  
            font-size: 0.8rem;  
            font-weight: 600;  
            border: 1px solid rgba(16, 185, 129, 0.3);  
        }  
        .badge-blockchain {  
            display: flex;  
            align-items: center;  
            gap: 0.4rem;  
            background: rgba(139, 92, 246, 0.12);  
            color: var(--accent-purple);  
            padding: 0.4rem 0.8rem;  
            border-radius: 9999px;  
            font-size: 0.8rem;  
            font-weight: 600;  
            border: 1px solid rgba(139, 92, 246, 0.3);  
        }  
        .badge-quantum {  
            display: flex;  
            align-items: center;  
            gap: 0.4rem;  
            background: rgba(6, 182, 212, 0.12);  
            color: var(--accent-cyan);  
            padding: 0.4rem 0.8rem;  
            border-radius: 9999px;  
            font-size: 0.8rem;  
            font-weight: 600;  
            border: 1px solid rgba(6, 182, 212, 0.3);  
        }  
        .badge-pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--accent-green); animation: pulse 1.5s infinite; }  
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }  

        .dashboard-layout {  
            display: grid;  
            grid-template-columns: 1fr 450px;  
            gap: 1.25rem;  
            max-width: 1540px;  
            width: 100%;  
        }  
        @media (max-width: 1200px) { .dashboard-layout { grid-template-columns: 1fr; } }  

        .main-viewport { display: flex; flex-direction: column; gap: 1rem; }  
        .video-container {  
            position: relative;  
            background: #000;  
            border-radius: 12px;  
            overflow: hidden;  
            box-shadow: 0 10px 30px rgba(0,0,0,0.6);  
            border: 1px solid var(--surface-border);  
        }  
        video { width: 100%; aspect-ratio: 16 / 9; display: block; object-fit: contain; }  

        .hud-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.65rem; }  
        .hud-card {  
            background: var(--surface);  
            border: 1px solid var(--surface-border);  
            padding: 0.75rem;  
            border-radius: 10px;  
            text-align: center;  
        }  
        .hud-val { font-size: 1.25rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: var(--primary); }  
        .hud-label { font-size: 0.70rem; color: var(--text-muted); text-transform: uppercase; margin-top: 0.2rem; }  

        .sidebar-panel { display: flex; flex-direction: column; gap: 1rem; }  
        .panel-card {  
            background: var(--surface);  
            border: 1px solid var(--surface-border);  
            border-radius: 12px;  
            padding: 1.1rem;  
        }  
        .panel-title {  
            font-size: 0.92rem;  
            font-weight: 700;  
            margin-bottom: 0.85rem;  
            display: flex;  
            justify-content: space-between;  
            align-items: center;  
        }  

        .btn-primary {  
            width: 100%;  
            padding: 0.70rem;  
            font-size: 0.88rem;  
            font-weight: 600;  
            border-radius: 8px;  
            border: none;  
            background: var(--primary);  
            color: #fff;  
            cursor: pointer;  
            transition: 0.2s ease;  
        }  
        .btn-primary:hover { opacity: 0.9; }  
        .btn-secondary { background: #1e293b; color: #e2e8f0; margin-top: 0.5rem; }  
        .btn-sos { background: #ef4444; color: #fff; font-weight: 700; letter-spacing: 0.05em; }  
        .btn-quantum { background: linear-gradient(135deg, #06b6d4, #8b5cf6); color: #fff; font-weight: 700; }  

        .control-row { display: flex; flex-direction: column; gap: 0.35rem; margin-bottom: 0.75rem; }  
        .control-row label { font-size: 0.82rem; color: var(--text-muted); display: flex; justify-content: space-between; }  
        input[type="range"] { width: 100%; accent-color: var(--accent-cyan); }  
        input[type="text"], textarea, select {  
            width: 100%;  
            padding: 0.55rem 0.75rem;  
            background: #090d16;  
            border: 1px solid var(--surface-border);  
            border-radius: 6px;  
            color: #fff;  
            font-family: 'Space Grotesk', sans-serif;  
            font-size: 0.85rem;  
        }  

        .upload-box {  
            border: 2px dashed var(--surface-border);  
            border-radius: 8px;  
            padding: 0.75rem;  
            text-align: center;  
            cursor: pointer;  
            background: rgba(255,255,255,0.02);  
            margin-bottom: 0.8rem;  
        }  
        .upload-box:hover { border-color: var(--primary); }  

        .violation-table {  
            font-family: 'JetBrains Mono', monospace;  
            font-size: 0.75rem;  
            max-height: 200px;  
            overflow-y: auto;  
            display: flex;  
            flex-direction: column;  
            gap: 0.4rem;  
        }  
        .violation-item {  
            background: rgba(239, 68, 68, 0.08);  
            padding: 0.5rem 0.7rem;  
            border-radius: 6px;  
            display: flex;  
            justify-content: space-between;  
            align-items: center;  
            border-left: 3px solid var(--accent-red);  
            text-decoration: none;  
            color: inherit;  
        }  
        .violation-item:hover { background: rgba(239, 68, 68, 0.16); }  
        .fine-tag { color: var(--accent-gold); font-weight: 700; font-size: 0.8rem; }  
        .speed-tag { color: var(--accent-red); font-weight: 700; font-size: 0.85rem; }  

        .vahan-result-card {  
            background: rgba(37, 99, 235, 0.08);  
            border: 1px solid rgba(37, 99, 235, 0.25);  
            padding: 0.75rem;  
            border-radius: 8px;  
            font-family: 'JetBrains Mono', monospace;  
            font-size: 0.76rem;  
            margin-top: 0.6rem;  
            display: none;  
        }  
        .status-badge {  
            font-family: 'JetBrains Mono', monospace;  
            font-size: 0.75rem;  
            padding: 0.35rem 0.6rem;  
            border-radius: 6px;  
            background: rgba(16, 185, 129, 0.1);  
            color: var(--accent-green);  
            border: 1px solid rgba(16, 185, 129, 0.25);  
            margin-bottom: 0.5rem;  
        }  

        .risk-gauge {  
            height: 10px;  
            border-radius: 5px;  
            background: #1e293b;  
            overflow: hidden;  
            margin-top: 0.3rem;  
        }  
        .risk-fill {  
            height: 100%;  
            width: 5%;  
            background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444);  
            transition: width 0.3s ease;  
        }  
        .detection-panel { margin-top: 0.9rem; }
        .detection-summary {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-bottom: 0.7rem;
        }
        .detection-chip {
            border: 1px solid var(--surface-border);
            border-radius: 999px;
            padding: 0.25rem 0.6rem;
            color: var(--accent-cyan);
            font: 0.72rem 'JetBrains Mono', monospace;
        }
        .detection-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
            gap: 0.45rem;
            max-height: 250px;
            overflow-y: auto;
        }
        .detection-item {
            display: flex;
            justify-content: space-between;
            gap: 0.6rem;
            padding: 0.55rem 0.65rem;
            background: rgba(255,255,255,0.025);
            border: 1px solid var(--surface-border);
            border-radius: 7px;
            font: 0.72rem 'JetBrains Mono', monospace;
        }
        .detection-item small { color: var(--text-muted); }
        .header-title { flex: 1 1 32rem; min-width: 0; }
        .header-title h1 { line-height: 1.25; overflow-wrap: anywhere; }
        .header-title p { line-height: 1.5; }
        .header-badges { justify-content: flex-end; }
        .dashboard-layout { grid-template-columns: minmax(0, 1fr) minmax(340px, 450px); }
        .main-viewport, .sidebar-panel, .panel-card, .system-overview > * { min-width: 0; }
        .system-overview { grid-template-columns: repeat(4, minmax(0, 1fr)); }
        .overview-value, .overview-delta, .hud-val { overflow-wrap: anywhere; }
        .hud-grid { grid-template-columns: repeat(5, minmax(0, 1fr)); }
        .panel-title { gap: 0.75rem; }
        .panel-title > * { min-width: 0; }
        .weather-summary {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem 1.25rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
        }
        .action-row { display: flex; flex-wrap: wrap; align-items: stretch; gap: 0.5rem; }
        .action-row > .btn-primary { flex: 1 1 9rem; width: auto; min-width: 0; }
        .registry-search { flex-wrap: nowrap; }
        .registry-search input { flex: 1 1 12rem; min-width: 0; }
        .registry-search .btn-primary { flex: 0 0 auto; }
        .quantum-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 0.5rem;
        }
        .quantum-grid > * { min-width: 0; }
        .detection-item { min-width: 0; overflow-wrap: anywhere; }
        .violation-item { gap: 0.5rem; flex-wrap: wrap; overflow-wrap: anywhere; }

        @media (max-width: 1200px) {
            .dashboard-layout { grid-template-columns: minmax(0, 1fr); }
            .sidebar-panel { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); align-items: start; }
        }
        @media (max-width: 900px) {
            .system-overview { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .hud-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            .quantum-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 760px) {
            body { padding: 0.75rem; }
            .header { align-items: flex-start; flex-direction: column; gap: 0.75rem; }
            .header-title { flex-basis: auto; width: 100%; }
            .header-title h1 { font-size: 1.1rem; }
            .header-badges { justify-content: flex-start; }
            .sidebar-panel { grid-template-columns: minmax(0, 1fr); }
            .weather-summary { align-items: flex-start; flex-direction: column; }
            .panel-card { padding: 0.9rem; }
        }
        @media (max-width: 520px) {
            .system-overview, .quantum-grid { grid-template-columns: minmax(0, 1fr); }
            .hud-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .header-badges { align-items: flex-start; flex-direction: column; }
            .panel-title { align-items: flex-start; flex-direction: column; }
            .action-row, .registry-search { flex-direction: column; }
            .action-row > .btn-primary, .registry-search .btn-primary { width: 100%; }
            .registry-search input { width: 100%; }
            .control-row label { gap: 0.4rem; }
        }
    </style>  
</head>  
<body>  
    <div class="header">  
        <div class="header-title">  
            <h1>Real-Time Multi-Hazard Traffic Surveillance and Automated e-Challan System using Edge YOLO and WebRTC</h1>  
            <p>Smart Highway Enforcement, AI Predictive Risk Assessment, Adaptive Signal Control, Green Corridor, Blockchain Ledger & Quantum Optimization</p>  
        </div>  
        <div class="header-badges">  
            <a class="api-explorer-link" href="#" onclick="window.open(window.TRAFFIC_API_INTERFACE_URL, '_blank', 'noopener'); return false;">API EXPLORER</a>
            <div class="badge-quantum" id="quantumStatusBadge">  
                ⚛️ QUANTUM: ACTIVE (QAOA • GROVER • QSVM)  
            </div>  
            <div class="badge-radar" id="databaseStatusBadge">DATABASE: CHECKING</div>
            <div class="badge-blockchain" id="blockchainStatusBadge">  
                ⛓️ BLOCKCHAIN: VERIFIED (1 BLOCKS)  
            </div>  
            <div class="badge-radar">  
                <div class="badge-pulse"></div>  
                SYSTEM STATUS: ONLINE  
            </div>  
        </div>  
    </div>  

    <div class="system-overview">
        <div class="overview-card">
            <div class="overview-label">AI Stack</div>
            <div class="overview-value" id="overviewAiStack">YOLO / Edge Vision</div>
            <div class="overview-delta" id="overviewAiStatus">Operational</div>
        </div>
        <div class="overview-card">
            <div class="overview-label">Signal Control</div>
            <div class="overview-value" id="overviewSignalMode">Adaptive</div>
            <div class="overview-delta" id="overviewSignalStatus">Green cycle online</div>
        </div>
        <div class="overview-card">
            <div class="overview-label">Risk Envelope</div>
            <div class="overview-value" id="overviewRiskLevel">NORMAL</div>
            <div class="overview-delta" id="overviewRiskStatus">Predictive engine stable</div>
        </div>
        <div class="overview-card">
            <div class="overview-label">Dispatch Network</div>
            <div class="overview-value" id="overviewDispatchMode">SANDBOX</div>
            <div class="overview-delta" id="overviewDispatchStatus">Blockchain ledger synced</div>
        </div>
    </div>

    <div class="dashboard-layout">  
        <div class="main-viewport">  
            <div class="video-container">  
                <video id="streamVideo" autoplay playsinline muted></video>  
            </div>  

            <div class="panel-card detection-panel">
                <div class="panel-title">
                    <span>Live Object Detection</span>
                    <span id="detectionModelLabel" style="color: var(--text-muted); font-size: 0.72rem;">Waiting for model</span>
                </div>
                <div class="detection-summary" id="detectionSummary">
                    <span class="overview-delta">Waiting for detector telemetry...</span>
                </div>
                <div class="detection-list" id="detectionList" aria-live="polite">
                    <div class="overview-delta">No detections received yet.</div>
                </div>
                <p style="font-size: 0.72rem; color: var(--text-muted); margin: 0.65rem 0 0;">
                    Labels and confidence scores depend on the loaded model and video quality; detections are advisory, not identity verification.
                </p>
            </div>

            <div class="hud-grid">  
                <div class="hud-card">  
                    <div class="hud-val" id="valFps">0.0</div>  
                    <div class="hud-label">FPS / Latency</div>  
                </div>  
                <div class="hud-card">  
                    <div class="hud-val" id="valPassed" style="color: var(--accent-cyan);">0</div>  
                    <div class="hud-label">Vehicles Scanned</div>  
                </div>  
                <div class="hud-card">  
                    <div class="hud-val" id="valRisk" style="color: var(--accent-green);">0%</div>  
                    <div class="hud-label">Predictive Risk</div>  
                </div>  
                <div class="hud-card">  
                    <div class="hud-val" id="valSignal" style="color: var(--accent-green);">GREEN</div>  
                    <div class="hud-label">Smart Signal</div>  
                </div>  
                <div class="hud-card">  
                    <div class="hud-val" id="valViolations" style="color: var(--accent-gold);">0</div>  
                    <div class="hud-label">Blockchain Challans</div>  
                </div>  
            </div>  

            <div class="panel-card">  
                <div class="panel-title">  
                    <span>AI Accident Risk Index & Predictive Safety Analysis</span>  
                    <span id="riskScoreLabel" style="font-family:'JetBrains Mono'; color:var(--accent-green);">5% (NORMAL)</span>  
                </div>  
                <div class="risk-gauge">  
                    <div class="risk-fill" id="riskFillBar"></div>  
                </div>  
                <p style="font-size: 0.78rem; color: var(--text-muted); margin: 0.4rem 0 0;">  
                    Predictive collision forecasting actively evaluates Time-To-Collision (TTC), erratic lateral lane drift, and aggressive tailgating.  
                </p>  
            </div>  

            <div class="panel-card weather-summary" style="display: flex; justify-content: space-between; align-items: center; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">  
                <div>  
                    <strong>WEATHER VISIBILITY:</strong> <span id="weatherText" style="color: var(--accent-cyan);">95% (Clear)</span>  
                </div>  
                <div>  
                    <strong>ADVISORY LIMIT:</strong> <span id="advisoryLimitText" style="color: var(--accent-gold);">60 km/h</span>  
                </div>  
                <div>  
                    <strong>CORRIDOR CO2:</strong> <span id="co2Text" style="color: var(--accent-green);">0.0 kg</span>  
                </div>  
            </div>  

            <!-- Quantum Computing Integration Module Dashboard Card -->
            <div class="panel-card" style="border: 1px solid rgba(6, 182, 212, 0.35); background: linear-gradient(180deg, #0e1320 0%, #08101e 100%);">  
                <div class="panel-title">  
                    <span style="color: var(--accent-cyan); display: flex; align-items: center; gap: 0.4rem;">  
                        ⚛️ Quantum Optimization, Fast Search & Quantum ML Modules  
                    </span>  
                    <span style="font-size: 0.75rem; color: var(--accent-purple); font-family: 'JetBrains Mono';">Hybrid Quantum-Edge Suite</span>  
                </div>  
                <div class="quantum-grid">  
                    <div style="background: rgba(6,182,212,0.06); padding: 0.75rem; border-radius: 8px; border: 1px solid rgba(6,182,212,0.2);">  
                        <div style="font-weight: 700; font-size: 0.82rem; color: var(--accent-cyan); margin-bottom: 0.3rem;">QAOA Optimization</div>  
                        <p style="font-size: 0.72rem; color: var(--text-muted); margin: 0 0 0.5rem;">Minimizes corridor queue & optimizes highway patrol enforcement.</p>  
                        <button class="btn-primary btn-quantum" style="font-size: 0.75rem; padding: 0.4rem;" onclick="runQAOAOptimization()">Run QAOA</button>  
                    </div>  
                    <div style="background: rgba(139,92,246,0.06); padding: 0.75rem; border-radius: 8px; border: 1px solid rgba(139,92,246,0.2);">  
                        <div style="font-weight: 700; font-size: 0.82rem; color: var(--accent-purple); margin-bottom: 0.3rem;">Grover Database Search</div>  
                        <p style="font-size: 0.72rem; color: var(--text-muted); margin: 0 0 0.5rem;">O(√N) quadratic speedup lookup across VAHAN & CCTNS.</p>  
                        <button class="btn-primary" style="background: var(--accent-purple); font-size: 0.75rem; padding: 0.4rem;" onclick="runGroverSearchPrompt()">Run Grover</button>  
                    </div>  
                    <div style="background: rgba(16,185,129,0.06); padding: 0.75rem; border-radius: 8px; border: 1px solid rgba(16,185,129,0.2);">  
                        <div style="font-weight: 700; font-size: 0.82rem; color: var(--accent-green); margin-bottom: 0.3rem;">QSVM Hazard Classifier</div>  
                        <p style="font-size: 0.72rem; color: var(--text-muted); margin: 0 0 0.5rem;">Quantum Feature Map kernel evaluation & HHL linear solver.</p>  
                        <button class="btn-primary" style="background: var(--accent-green); font-size: 0.75rem; padding: 0.4rem;" onclick="runQSVMHazardClassification()">Run QSVM</button>  
                    </div>  
                </div>  
                <div id="quantumOutputCard" style="display:none; margin-top: 0.75rem; padding: 0.7rem; border-radius: 6px; background: #070b14; border: 1px solid var(--surface-border); font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;"></div>  
            </div>  
        </div>  

        <div class="sidebar-panel">  
            <div class="panel-card">  
                <div class="panel-title">Video Feed & Live Stream</div>  
                <div class="status-badge" id="rtoStatusText">e-Challan Gateway: Connected</div>  
                <div class="upload-box" onclick="document.getElementById('videoFileInput').click()">  
                    <span id="uploadStatusText">Upload Video File (MP4/AVI)</span>  
                    <input type="file" id="videoFileInput" accept="video/*" style="display:none" onchange="uploadVideoFile()">  
                </div>  
                <div class="action-row">  
                    <button class="btn-primary" style="font-size:0.85rem;" onclick="switchSourceMode('0')">Use Camera</button>  
                    <button class="btn-primary btn-secondary" style="font-size:0.85rem; margin-top:0;" onclick="switchSourceMode('sim')">Use Simulator</button>  
                </div>  
                <button id="streamBtn" class="btn-primary" style="margin-top:0.6rem;" onclick="toggleStream()">Start WebRTC Radar Stream</button>  
            </div>  

            <div class="panel-card">  
                <div class="panel-title">Adaptive Smart Signal Controller</div>  
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; margin-bottom: 0.5rem;">  
                    <span>Adaptive Density Timing:</span>  
                    <strong style="color: var(--accent-cyan);" id="adaptiveModeText">ENABLED</strong>  
                </div>  
                <div class="action-row">  
                    <button class="btn-primary" style="font-size: 0.75rem; padding: 0.4rem;" onclick="setSignalOverride('GREEN')">Force Green</button>  
                    <button class="btn-primary" style="font-size: 0.75rem; padding: 0.4rem; background: var(--accent-gold);" onclick="setSignalOverride('YELLOW')">Force Yellow</button>  
                    <button class="btn-primary" style="font-size: 0.75rem; padding: 0.4rem; background: var(--accent-red);" onclick="setSignalOverride('RED')">Force Red</button>  
                    <button class="btn-primary btn-secondary" style="font-size: 0.75rem; padding: 0.4rem; margin-top:0;" onclick="toggleAdaptiveSignal()">Auto</button>  
                </div>  
                <div class="action-row" style="margin-top: 0.5rem;">  
                    <button class="btn-primary" style="font-size: 0.75rem; padding: 0.4rem; background: #059669;" onclick="triggerGreenCorridor()">🚑 Trigger Green Corridor</button>  
                    <button class="btn-primary btn-secondary" style="font-size: 0.75rem; padding: 0.4rem; margin-top:0;" onclick="cancelGreenCorridor()">Cancel</button>  
                </div>  
            </div>  

            <div class="panel-card">  
                <div class="panel-title">Citizen Safety & Community Report</div>  
                <div style="display: flex; flex-direction: column; gap: 0.5rem;">  
                    <input type="text" id="citizenLocationInput" placeholder="Location (e.g. NH-44 Tirupati Bypass)">  
                    <input type="text" id="citizenDescInput" placeholder="Hazard or Reckless Driving Description">  
                    <div class="action-row">  
                        <button class="btn-primary" style="font-size: 0.8rem; padding: 0.5rem;" onclick="submitCitizenReport()">Submit Report</button>  
                        <button class="btn-primary btn-sos" style="font-size: 0.8rem; padding: 0.5rem; width: auto;" onclick="triggerEmergencySOS()">SOS</button>  
                    </div>  
                </div>  
            </div>  

            <div class="panel-card">  
                <div class="panel-title">VAHAN Registry Query & HSRP Audit</div>  
                <div class="action-row registry-search">  
                    <input type="text" id="vahanSearchInput" placeholder="e.g. AP 39 MN 3019 or Tirupati">  
                    <button class="btn-primary" style="width: auto; padding: 0.5rem 0.9rem;" onclick="searchVahanRegistry()">Search</button>  
                </div>  
                <div class="vahan-result-card" id="vahanResultBox"></div>  
            </div>  

            <div class="panel-card">  
                <div class="panel-title">Speed Limit & Laser Calibration</div>  
                <div class="control-row">
                    <label><span>Detection Confidence Threshold</span><span id="confLabel">30%</span></label>
                    <input type="range" id="confRange" min="15" max="85" step="5" value="30" oninput="pushSettings()">
                </div>
                <div class="control-row">  
                    <label><span>Speed Limit</span><span id="limitLabel">60 km/h</span></label>  
                    <input type="range" id="limitRange" min="30" max="150" step="5" value="60" oninput="pushSettings()">  
                </div>  
                <div class="control-row">  
                    <label><span>Calibration Distance</span><span id="distLabel">20 m</span></label>  
                    <input type="range" id="distRange" min="5" max="60" step="1" value="20" oninput="pushSettings()">  
                </div>  
                <div class="control-row">  
                    <label><span>Trap Line 1 (Y)</span><span id="l1Label">240 px</span></label>  
                    <input type="range" id="l1Range" min="80" max="450" step="10" value="240" oninput="pushSettings()">  
                </div>  
                <div class="control-row">  
                    <label><span>Trap Line 2 (Y)</span><span id="l2Label">520 px</span></label>  
                    <input type="range" id="l2Range" min="400" max="680" step="10" value="520" oninput="pushSettings()">  
                </div>  
            </div>  

            <div class="panel-card" style="flex-grow: 1;">  
                <div class="panel-title">  
                    <span>Live Blockchain e-Challan Dispatch</span>  
                    <button class="btn-primary" style="width: auto; padding: 0.3rem 0.6rem; font-size: 0.72rem;" onclick="exportCSV()">Export CSV</button>  
                </div>  
                <div class="violation-table" id="violationLogBox">  
                    <div style="color: var(--text-muted); padding: 0.5rem;">Scanning highway traffic corridor...</div>  
                </div>  
            </div>  
        </div>  
    </div>  

    <script>  
        let pc = null;  
        let ws = null;  
        let dataChannel = null;  

        async function refreshDatabaseStatus() {
            const badge = document.getElementById('databaseStatusBadge');
            try {
                const response = await fetch('/api/health');
                if (!response.ok) throw new Error(`Database health returned HTTP ${response.status}`);
                const health = await response.json();
                badge.innerText = `DATABASE: ${health.database.toUpperCase()} ONLINE`;
                badge.style.color = 'var(--accent-green)';
            } catch (error) {
                badge.innerText = 'DATABASE: UNAVAILABLE';
                badge.style.color = 'var(--accent-red)';
                console.error('Database health check failed', error);
            }
        }

        async function refreshSystemOverview() {
            try {
                const response = await fetch('/api/system/overview');
                if (!response.ok) throw new Error('Overview request failed');
                const overview = await response.json();
                document.getElementById('overviewAiStack').innerText = 'YOLO / Edge Vision';
                document.getElementById('overviewAiStatus').innerText = overview.status === 'operational' ? 'Operational' : 'Degraded';
                document.getElementById('overviewSignalMode').innerText = overview.system_summary?.adaptive_signal ? 'Adaptive' : 'Manual';
                document.getElementById('overviewSignalStatus').innerText = overview.system_summary?.adaptive_signal ? 'Green cycle online' : 'Override enabled';
                const riskLevel = document.getElementById('valRisk').innerText || '0%';
                document.getElementById('overviewRiskLevel').innerText = riskLevel;
                document.getElementById('overviewRiskStatus').innerText = riskLevel.includes('70') || riskLevel.includes('80') || riskLevel.includes('90') ? 'Alert window active' : 'Predictive engine stable';
                document.getElementById('overviewDispatchMode').innerText = overview.database ? overview.database.toUpperCase() : 'DATABASE';
                document.getElementById('overviewDispatchStatus').innerText = overview.generated_at ? 'Synced at ' + overview.generated_at : 'Blockchain ledger synced';
            } catch (error) {
                console.error('System overview refresh failed', error);
            }
        }
        refreshDatabaseStatus();
        refreshSystemOverview();
        window.setInterval(refreshDatabaseStatus, 30000);
        window.setInterval(refreshSystemOverview, 15000);

        function renderDetections(data) {
            const summary = document.getElementById('detectionSummary');
            const list = document.getElementById('detectionList');
            const detections = Array.isArray(data.detections) ? data.detections : [];
            const counts = data.object_counts && typeof data.object_counts === 'object' ? data.object_counts : {};
            document.getElementById('detectionModelLabel').textContent =
                `Model: ${data.detection_model || 'unknown'} · ${detections.length} visible`;

            summary.replaceChildren();
            const countEntries = Object.entries(counts);
            if (!countEntries.length) {
                const emptySummary = document.createElement('span');
                emptySummary.className = 'overview-delta';
                emptySummary.textContent = 'No objects detected in the latest frame.';
                summary.appendChild(emptySummary);
            } else {
                countEntries.forEach(([label, count]) => {
                    const chip = document.createElement('span');
                    chip.className = 'detection-chip';
                    chip.textContent = `${label} × ${count}`;
                    summary.appendChild(chip);
                });
            }

            list.replaceChildren();
            if (!detections.length) {
                const emptyList = document.createElement('div');
                emptyList.className = 'overview-delta';
                emptyList.textContent = 'No detections received in the latest frame.';
                list.appendChild(emptyList);
                return;
            }

            detections.slice(0, 30).forEach((detection) => {
                const item = document.createElement('div');
                item.className = 'detection-item';
                const identity = document.createElement('span');
                identity.textContent = `${detection.label || 'Unknown'} · #${detection.track_id ?? '?'}`;
                const score = document.createElement('small');
                const confidence = Number(detection.confidence);
                score.textContent = Number.isFinite(confidence) ? `${Math.round(confidence * 100)}%` : 'N/A';
                item.append(identity, score);
                list.appendChild(item);
            });
        }

        function applyTelemetry(data) {  
            if (!data) return;  
            document.getElementById('valFps').innerText = (data.fps || 0).toFixed(1);  
            document.getElementById('valPassed').innerText = data.total_passed || 0;  
            document.getElementById('valViolations').innerText = data.total_violations || 0;  
            renderDetections(data);

            const risk = data.predictive_risk_score || 0;  
            document.getElementById('overviewRiskLevel').innerText = `${risk.toFixed(0)}%`;
            document.getElementById('overviewRiskStatus').innerText =
                risk > 70 ? 'High risk · review alerts' : risk > 40 ? 'Elevated risk' : 'Predictive engine stable';
            document.getElementById('valRisk').innerText = risk.toFixed(0) + "%";  
            document.getElementById('riskFillBar').style.width = Math.max(5, risk) + "%";  
            const riskLabel = document.getElementById('riskScoreLabel');  
            if (risk > 70) {  
                riskLabel.innerText = risk + "% (CRITICAL RISK)";  
                riskLabel.style.color = "var(--accent-red)";  
                document.getElementById('valRisk').style.color = "var(--accent-red)";  
            } else if (risk > 40) {  
                riskLabel.innerText = risk + "% (ELEVATED)";  
                riskLabel.style.color = "var(--accent-gold)";  
                document.getElementById('valRisk').style.color = "var(--accent-gold)";  
            } else {  
                riskLabel.innerText = risk + "% (NORMAL)";  
                riskLabel.style.color = "var(--accent-green)";  
                document.getElementById('valRisk').style.color = "var(--accent-green)";  
            }  

            if (data.traffic_signal) {  
                const sigVal = document.getElementById('valSignal');  
                if (data.green_corridor && data.green_corridor.active) {  
                    sigVal.innerText = "EMERGENCY GREEN";  
                    sigVal.style.color = "#10b981";  
                    document.getElementById('adaptiveModeText').innerText = "GREEN CORRIDOR ACTIVE";  
                } else {  
                    sigVal.innerText = data.traffic_signal.phase;  
                    sigVal.style.color = data.traffic_signal.phase === "GREEN" ? "var(--accent-green)" : (data.traffic_signal.phase === "YELLOW" ? "var(--accent-gold)" : "var(--accent-red)");  
                    document.getElementById('adaptiveModeText').innerText = data.traffic_signal.adaptive_active ? "ENABLED" : "MANUAL OVERRIDE";  
                }  
            }  

            if (data.weather) {  
                document.getElementById('weatherText').innerText = `${data.weather.visibility_pct}% (${data.weather.weather_condition || data.weather.condition})`;  
                document.getElementById('advisoryLimitText').innerText = `${data.weather.advisory_speed_kmh} km/h`;  
            }  

            if (data.carbon_emissions) {  
                document.getElementById('co2Text').innerText = `${data.carbon_emissions.total_co2_kg} kg`;  
            }  

            if (data.blockchain) {  
                document.getElementById('blockchainStatusBadge').innerText = `⛓️ BLOCKCHAIN: ${data.blockchain.verified ? 'VERIFIED' : 'TAMPERED'} (${data.blockchain.total_blocks} BLOCKS)`;  
            }  

            if (data.rto_gateway_status) {  
                document.getElementById('rtoStatusText').innerText = "Gateway: " + data.rto_gateway_status;  
            }  

            if (data.recent_violations && data.recent_violations.length > 0) {  
                const logBox = document.getElementById('violationLogBox');  
                logBox.innerHTML = data.recent_violations.map(v =>   
                    `<a href="${v.snapshot_url}" target="_blank" class="violation-item" style="${v.cctns_alert ? 'border-left-color: #f59e0b; background: rgba(245,158,11,0.1);' : ''}">  
                        <div>  
                            <strong>[${v.challan_no}] ${v.plate}</strong> ${v.cctns_alert ? '<span style="color: #f59e0b;">⚠️ HOTLIST</span>' : ''}<br>  
                            <span style="color: var(--text-muted)">${v.owner} | Block #${v.blockchain_block} | ${v.violation}</span>  
                        </div>  
                        <div style="text-align: right;">  
                            <div class="speed-tag">${v.speed} km/h</div>  
                            <div class="fine-tag">Rs. ${v.fine}</div>  
                        </div>  
                    </a>`  
                ).join('');  
            }  
        }  

        let telemetryFramePending = false;
        let latestTelemetry = null;
        function queueTelemetry(data) {
            latestTelemetry = data;
            if (telemetryFramePending) return;
            telemetryFramePending = true;
            requestAnimationFrame(() => {
                telemetryFramePending = false;
                const current = latestTelemetry;
                latestTelemetry = null;
                if (current) applyTelemetry(current);
            });
        }

        function initWebSocket() {  
            try {  
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';  
                ws = new WebSocket(protocol + '//' + window.location.host + '/ws/telemetry');  
                ws.onmessage = (event) => {  
                    try {
                        const message = JSON.parse(event.data);
                        if (!message.error) queueTelemetry(message);
                    } catch (error) {
                        console.error('Invalid telemetry message', error);
                    }
                };  
                ws.onclose = () => { setTimeout(initWebSocket, 4000); };  
                ws.onerror = () => { ws.close(); };  
            } catch (error) {
                console.error('WebSocket initialization failed', error);
                window.setTimeout(initWebSocket, 4000);
            }
        }  
        initWebSocket();  

        async function toggleStream() {  
            const btn = document.getElementById('streamBtn');  
            if (pc) {  
                pc.close();  
                pc = null;  
                dataChannel = null;
                document.getElementById('streamVideo').srcObject = null;
                btn.innerText = "Start WebRTC Radar Stream";  
                btn.style.background = "var(--primary)";  
                return;  
            }  

            btn.disabled = true;  
            btn.innerText = "Connecting WebRTC...";  

            try {
                pc = new RTCPeerConnection({
                    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
                });
                const connection = pc;
                dataChannel = connection.createDataChannel('telemetry');

                connection.ontrack = (event) => {
                    const video = document.getElementById('streamVideo');
                    video.srcObject = event.streams && event.streams[0]
                        ? event.streams[0]
                        : new MediaStream([event.track]);
                };

                connection.addTransceiver('video', { direction: 'recvonly' });
                const offer = await connection.createOffer();
                await connection.setLocalDescription(offer);

                await new Promise((resolve, reject) => {
                    if (connection.iceGatheringState === 'complete') {
                        resolve();
                        return;
                    }
                    const timeout = window.setTimeout(() => {
                        connection.removeEventListener('icegatheringstatechange', check);
                        reject(new Error('Timed out gathering WebRTC connection candidates.'));
                    }, 10000);
                    const check = () => {
                        if (connection.iceGatheringState === 'complete') {
                            window.clearTimeout(timeout);
                            connection.removeEventListener('icegatheringstatechange', check);
                            resolve();
                        }
                    };
                    connection.addEventListener('icegatheringstatechange', check);
                    check();
                });

                const resp = await fetch('/offer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sdp: connection.localDescription.sdp,
                        type: connection.localDescription.type
                    })
                });
                if (!resp.ok) throw new Error(`WebRTC signaling failed with HTTP ${resp.status}`);
                await connection.setRemoteDescription(await resp.json());

                btn.disabled = false;
                btn.innerText = "Stop Radar Stream";
                btn.style.background = "#ef4444";
            } catch (error) {
                if (pc) pc.close();
                pc = null;
                dataChannel = null;
                btn.disabled = false;
                btn.innerText = "Start WebRTC Radar Stream";
                btn.style.background = "var(--primary)";
                console.error('Unable to start WebRTC stream', error);
            }
        }  

        function pushSettings() {  
            const limit = parseFloat(document.getElementById('limitRange').value);  
            const dist = parseFloat(document.getElementById('distRange').value);  
            const confidence = parseInt(document.getElementById('confRange').value, 10) / 100;
            const l1 = parseInt(document.getElementById('l1Range').value);  
            const l2 = parseInt(document.getElementById('l2Range').value);  

            document.getElementById('limitLabel').innerText = limit + " km/h";  
            document.getElementById('distLabel').innerText = dist + " m";  
            document.getElementById('confLabel').innerText = Math.round(confidence * 100) + "%";
            document.getElementById('l1Label').innerText = l1 + " px";  
            document.getElementById('l2Label').innerText = l2 + " px";  

            const payload = JSON.stringify({  
                speed_limit: limit,  
                distance_m: dist,  
                conf: confidence,
                line1_y: l1,  
                line2_y: l2  
            });  

            if (ws && ws.readyState === WebSocket.OPEN) {  
                ws.send(payload);  
            } else if (dataChannel && dataChannel.readyState === 'open') {  
                dataChannel.send(payload);  
            }  
        }  

        async function uploadVideoFile() {  
            const input = document.getElementById('videoFileInput');  
            if (!input.files || input.files.length === 0) return;  

            const file = input.files[0];  
            const formData = new FormData();  
            formData.append('file', file);  
            document.getElementById('uploadStatusText').innerText = "Uploading " + file.name + "...";  

            const resp = await fetch('/api/upload_video', { method: 'POST', body: formData });  
            const res = await resp.json();  
            document.getElementById('uploadStatusText').innerText = res.status === 'ok' ? "Active: " + file.name : "Upload failed.";  
        }  

        async function switchSourceMode(mode) {  
            await fetch('/api/switch_source?source=' + encodeURIComponent(mode), { method: 'POST' });  
            document.getElementById('uploadStatusText').innerText = mode === '0' ? "Live Camera Active" : "Simulator Active";  
        }  

        async function searchVahanRegistry() {  
            const q = document.getElementById('vahanSearchInput').value.trim();  
            if (!q) return;  
            const resp = await fetch('/api/vahan_lookup?query=' + encodeURIComponent(q));  
            const res = await resp.json();  
            const box = document.getElementById('vahanResultBox');  
            if (res.status === 'found' && res.results.length > 0) {  
                const rec = res.results[0];  
                const hsrpResp = await fetch('/api/hsrp_check?plate=' + encodeURIComponent(rec.plate));  
                const hsrpData = await hsrpResp.json();  
                box.style.display = 'block';  
                box.innerHTML = `  
                    <div style="color: var(--accent-cyan); font-weight: 700;">Plate: ${rec.plate} [HSRP: ${hsrpData.is_hsrp_compliant ? '✅ COMPLIANT' : '⚠️ NON-COMPLIANT'}]</div>  
                    <div><strong>Owner:</strong> ${rec.owner_name}</div>  
                    <div><strong>Vehicle:</strong> ${rec.make_model} (${rec.fuel_type})</div>  
                    <div><strong>RTO Office:</strong> ${rec.dto_office} (${rec.district})</div>  
                    <div><strong>FASTag ID:</strong> ${rec.fastag_id}</div>  
                    <div><strong>Insurance:</strong> ${rec.insurance_status} | <strong>PUC:</strong> ${rec.puc_status}</div>  
                `;  
            } else {  
                box.style.display = 'block';  
                box.innerHTML = `<span style="color: var(--accent-red);">No matching record found in registry.</span>`;  
            }  
        }  

        async function setSignalOverride(phase) {  
            await fetch('/api/traffic_signal/override?phase=' + encodeURIComponent(phase), { method: 'POST' });  
        }  

        async function toggleAdaptiveSignal() {  
            await fetch('/api/traffic_signal/adaptive_toggle', { method: 'POST' });  
        }  

        async function triggerGreenCorridor() {  
            const resp = await fetch('/api/green_corridor/trigger', {  
                method: 'POST',  
                headers: { 'Content-Type': 'application/json' },  
                body: JSON.stringify({ emergency_type: "AMBULANCE", vehicle_id: "AP-108-EMERGENCY" })  
            });  
            const res = await resp.json();  
            alert("🚨 " + res.status + " | Emergency Preemption Active for 30s");  
        }  

        async function cancelGreenCorridor() {  
            await fetch('/api/green_corridor/cancel', { method: 'POST' });  
            alert("Green Corridor cancelled. System restored to normal adaptive cycle.");  
        }  

        async function submitCitizenReport() {  
            const loc = document.getElementById('citizenLocationInput').value.trim();  
            const desc = document.getElementById('citizenDescInput').value.trim();  
            if (!loc || !desc) {  
                alert("Please provide both location and description.");  
                return;  
            }  
            const resp = await fetch('/api/citizen/report_hazard', {  
                method: 'POST',  
                headers: { 'Content-Type': 'application/json' },  
                body: JSON.stringify({ location: loc, description: desc, report_type: "CITIZEN_HAZARD" })  
            });  
            const res = await resp.json();  
            alert("Citizen report submitted successfully! Tracking ID: " + res.report.report_id);  
            document.getElementById('citizenLocationInput').value = '';  
            document.getElementById('citizenDescInput').value = '';  
        }  

        async function triggerEmergencySOS() {  
            const loc = document.getElementById('citizenLocationInput').value.trim() || "Active Surveillance Zone (Live GPS)";  
            if (confirm("Send emergency SOS alert to nearest Highway Patrol & 112 Control Room?")) {  
                const resp = await fetch('/api/citizen/sos_alert', {  
                    method: 'POST',  
                    headers: { 'Content-Type': 'application/json' },  
                    body: JSON.stringify({ location: loc, details: "Citizen Emergency Request from Dashboard" })  
                });  
                const res = await resp.json();  
                alert("🚨 SOS Broadcast Confirmed! Reference ID: " + res.event.report_id);  
            }  
        }  

        function exportCSV() {  
            window.location.href = window.TRAFFIC_API_BASE + '/api/export_violations';
        }  

        // Quantum Functions
        async function runQAOAOptimization() {
            const outBox = document.getElementById('quantumOutputCard');
            outBox.style.display = 'block';
            outBox.innerHTML = '<span style="color:var(--accent-cyan);">⚛️ Running QAOA Quantum Circuit Optimization on Corridor Segments...</span>';
            const resp = await fetch('/api/quantum/qaoa_optimize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ corridor_congestion: [4.2, 8.5, 3.1, 7.8], p_layers: 2 })
            });
            const res = await resp.json();
            outBox.innerHTML = `
                <div style="color:var(--accent-cyan); font-weight:700;">⚛️ QAOA QUANTUM OPTIMIZATION RESULTS:</div>
                <div>Status: ${res.status} | Ground Energy: ${res.ground_energy}</div>
                <div>Optimal Bitstring: <strong>[${res.optimal_bitstring.join(', ')}]</strong> (State Index: ${res.optimal_state_index})</div>
                <div>Quantum Fidelity: <strong>${(res.quantum_fidelity * 100).toFixed(1)}%</strong></div>
                <div>Optimized Green Times: [${res.green_allocations_sec.join('s, ')}s]</div>
                <div>Priority Intercept Nodes: ${res.patrol_intercept_nodes.join(' | ')}</div>
            `;
        }

        async function runGroverSearchPrompt() {
            const plate = prompt("Enter target license plate to search using Grover's Algorithm:", "AP 03 BK 4921");
            if (!plate) return;
            const outBox = document.getElementById('quantumOutputCard');
            outBox.style.display = 'block';
            outBox.innerHTML = `<span style="color:var(--accent-purple);">⚛️ Executing Grover's Algorithm with O(√N) Quadratic Search on Database...</span>`;
            const resp = await fetch('/api/quantum/grover_search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ target_plate: plate })
            });
            const res = await resp.json();
            outBox.innerHTML = `
                <div style="color:var(--accent-purple); font-weight:700;">⚛️ GROVER'S QUANTUM SEARCH COMPLETE:</div>
                <div>Target Matched: <strong>${res.matched_record.plate}</strong> (${res.matched_record.owner_name})</div>
                <div>Database Space: N = ${res.database_size} | Quantum Iterations: ${res.quantum_iterations_required}</div>
                <div>Quantum Speedup: <strong>${res.theoretical_speedup}</strong> (vs ${res.classical_average_queries} classical queries)</div>
                <div>Measurement Probability: <strong>${(res.success_probability * 100).toFixed(1)}%</strong></div>
                <div>Registered DTO: ${res.matched_record.dto_office} | FASTag: ${res.matched_record.fastag_id}</div>
            `;
        }

        async function runQSVMHazardClassification() {
            const outBox = document.getElementById('quantumOutputCard');
            outBox.style.display = 'block';
            outBox.innerHTML = '<span style="color:var(--accent-green);">⚛️ Evaluating Quantum Kernel Map and HHL Linear System Solver...</span>';
            const resp = await fetch('/api/quantum/qsvm_hazard_classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ speed_kmh: 88.5, lateral_drift_px: 48.0, gap_m: 6.5, sound_db: 96.0 })
            });
            const res = await resp.json();
            outBox.innerHTML = `
                <div style="color:var(--accent-green); font-weight:700;">⚛️ QSVM QUANTUM HAZARD CLASSIFICATION:</div>
                <div>Hazard Prediction: <strong style="color:var(--accent-red);">${res.predicted_category}</strong></div>
                <div>Confidence Score: <strong>${(res.quantum_confidence * 100).toFixed(1)}%</strong></div>
                <div>Quantum Kernel Alignment: K(x, centroid) = ${res.quantum_kernel_similarity}</div>
                <div>HHL Linear System Solver: ${res.hhl_status} (Condition Number κ: ${res.matrix_condition_number})</div>
                <div>Enforcement Recommendation: ${res.recommended_action}</div>
            `;
        }
    </script>  
</body>  
</html>  
"""
    dashboard_bootstrap = f"""<script>
window.TRAFFIC_API_BASE = {json.dumps(api_url)} || window.location.origin;
window.TRAFFIC_API_INTERFACE_URL = {json.dumps(api_interface_url)};
const dashboardFetch = window.fetch.bind(window);
window.fetch = (input, init) => {{
    if (typeof input === 'string' && input.startsWith('/')) {{
        input = new URL(input, window.TRAFFIC_API_BASE).href;
    }}
    return dashboardFetch(input, init);
}};
const DashboardWebSocket = window.WebSocket;
window.WebSocket = class extends DashboardWebSocket {{
    constructor(url, protocols) {{
        const target = new URL(url, window.location.href);
        if (target.host === window.location.host) {{
            const api = new URL(window.TRAFFIC_API_BASE);
            target.protocol = api.protocol === 'https:' ? 'wss:' : 'ws:';
            target.host = api.host;
        }}
        if (protocols === undefined) {{
            super(target.href);
        }} else {{
            super(target.href, protocols);
        }}
    }}
}};
</script>"""
    dashboard_html = dashboard_html.replace("</head>", dashboard_bootstrap + "</head>", 1)
    return dashboard_html
