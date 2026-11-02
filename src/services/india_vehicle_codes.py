import csv
import re
from pathlib import Path
from typing import Dict, Optional, TypedDict


STATE_CODES: Dict[str, str] = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Dadra and Nagar Haveli and Daman and Diu",
    "DL": "Delhi",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TG": "Telangana",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "UK": "Uttarakhand",
    "WB": "West Bengal",
}

STATE_TYPES: Dict[str, str] = {
    code: "union_territory"
    for code in ("AN", "CH", "DD", "DL", "JK", "LA", "LD", "PY")
}
STATE_TYPES.update({code: "state" for code in STATE_CODES if code not in STATE_TYPES})

LEGACY_STATE_CODES: Dict[str, str] = {
    "DN": "DD",
    "OR": "OD",
    "TS": "TG",
    "UA": "UK",
}


class VehicleCodeInfo(TypedDict):
    state_code: str
    state_name: str
    state_type: str
    district_code: str
    rto_code: str
    rto_area: Optional[str]


def _load_rto_codes() -> Dict[str, str]:
    csv_path = Path(__file__).resolve().parents[2] / "data" / "india_rto_codes.csv"
    rto_codes: Dict[str, str] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            raw_code = row.get("RegNo", "").strip().upper()
            match = re.fullmatch(r"([A-Z]{2})(\d{2})", raw_code)
            if match is None:
                continue
            prefix, number = match.groups()
            canonical_prefix = LEGACY_STATE_CODES.get(prefix, prefix)
            if canonical_prefix not in STATE_CODES:
                continue
            area = row.get("Place", "").strip()
            if not area:
                continue
            code = f"{prefix}{number}"
            rto_codes[code] = area
            if prefix in ("OR", "UA"):
                rto_codes[f"{canonical_prefix}{number}"] = area
    return rto_codes


RTO_CODES = _load_rto_codes()


def lookup_indian_vehicle_code(raw_plate: str) -> Optional[VehicleCodeInfo]:
    normalized = re.sub(r"[\s-]", "", raw_plate).upper()
    match = re.match(r"^([A-Z]{2})(\d{2})", normalized)
    if match is None:
        return None

    raw_state_code, rto_number = match.groups()
    state_code = LEGACY_STATE_CODES.get(raw_state_code, raw_state_code)
    state_name = STATE_CODES.get(state_code)
    if state_name is None:
        return None

    rto_code = f"{raw_state_code}{rto_number}"
    return {
        "state_code": raw_state_code,
        "state_name": state_name,
        "state_type": STATE_TYPES[state_code],
        "district_code": f"{raw_state_code} {rto_number}",
        "rto_code": rto_code,
        "rto_area": RTO_CODES.get(rto_code) or RTO_CODES.get(f"{state_code}{rto_number}"),
    }
