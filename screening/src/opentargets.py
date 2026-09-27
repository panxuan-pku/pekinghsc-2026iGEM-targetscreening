"""Disease-specific Open Targets annotations; never part of the ranking score."""
from datetime import datetime, timezone
import json

import pandas as pd
import requests

ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"
QUERY = """query($gene: String!, $disease: String!, $diseases: [String!]!) {
  meta { dataVersion { year month iteration } }
  disease(efoId: $disease) { id name }
  target(ensemblId: $gene) {
    id
    associations: associatedDiseases(Bs: $diseases, enableIndirect: false,
                                     page: {index: 0, size: 100}) {
      count rows { disease { id name } score datasourceScores { id score } }
    }
    background: associatedDiseases(enableIndirect: false, page: {index: 0, size: 10}) {
      rows { disease { id name } score datasourceScores { id score } }
    }
  }
}"""


def fetch_snapshot(gene_ids, disease_id, phenotype_ids, release, post=None):
    if len(set([disease_id, *phenotype_ids])) > 100:
        raise ValueError("at most 100 disease/phenotype IDs per snapshot")
    post = post or requests.post
    snapshot = {"endpoint": ENDPOINT, "query": QUERY, "release": release,
                "disease_id": disease_id, "phenotype_ids": phenotype_ids,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(), "records": {}}
    for gid in gene_ids:
        variables = {"gene": gid, "disease": disease_id,
                     "diseases": list(dict.fromkeys([disease_id, *phenotype_ids]))}
        record = {"variables": variables}
        try:
            response = post(ENDPOINT, json={"query": QUERY, "variables": variables}, timeout=(15, 30))
            response.raise_for_status()
            payload = response.json()
            record["response"] = payload
            if payload.get("errors"):
                raise ValueError(json.dumps(payload["errors"], ensure_ascii=False))
            data = payload["data"]
            version = data["meta"]["dataVersion"]
            actual = f'{int(version["year"]):02d}.{int(version["month"]):02d}'
            if version.get("iteration") is not None:
                actual += f'.{version["iteration"]}'
            if actual != release:
                raise ValueError(f"OT release mismatch: expected {release}, received {actual}")
            if not data.get("disease") or data["disease"]["id"] != disease_id:
                raise ValueError("OT disease identity not found")
            target = data.get("target")
            if not target or target["id"] != gid:
                raise ValueError("OT target identity not found")
            associations = target["associations"]
            rows = associations["rows"]
            if associations["count"] != len(rows) or any(r["disease"]["id"] not in variables["diseases"] for r in rows):
                raise ValueError("OT filtered associations incomplete or unexpected")
            record.update(status="ok", associations=rows, background=target["background"]["rows"])
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            record.update(status="query_failed", error=str(exc))
        snapshot["records"][gid] = record
    return snapshot


def annotate_snapshot(frame, snapshot, disease_id, phenotype_ids, release):
    if (snapshot["release"] != release or snapshot["disease_id"] != disease_id
            or snapshot["phenotype_ids"] != phenotype_ids):
        raise ValueError("OT snapshot disease, phenotype list or release mismatch")
    rows = []
    for gid in frame.gene_id:
        record = snapshot["records"].get(gid)
        if record is None:
            raise ValueError(f"OT snapshot missing candidate: {gid}")
        row = {"gene_id": gid, "ot_disease_id": disease_id, "ot_release": release,
               "ot_status": "query_failed", "ot_disease_score": float("nan"),
               "ot_datasource_scores": "[]", "ot_phenotype_evidence": "[]",
               "ot_other_disease_background_top10": "[]", "ot_error": record.get("error", ""),
               "ot_url": f"https://platform.opentargets.org/target/{gid}/associations"}
        if record["status"] == "ok":
            associations = record["associations"]
            direct = [r for r in associations if r["disease"]["id"] == disease_id]
            if len(direct) > 1:
                raise ValueError(f"duplicate OT disease association: {gid}")
            row["ot_status"] = "available" if direct else "no_record"
            if direct:
                score = direct[0]["score"]
                if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                    raise ValueError(f"invalid OT score: {gid}")
                row["ot_disease_score"] = score
                row["ot_datasource_scores"] = json.dumps(direct[0]["datasourceScores"], ensure_ascii=False)
            row["ot_phenotype_evidence"] = json.dumps([r for r in associations if r["disease"]["id"] in phenotype_ids], ensure_ascii=False)
            row["ot_other_disease_background_top10"] = json.dumps([
                r for r in record.get("background", []) if r["disease"]["id"] not in {disease_id, *phenotype_ids}
            ], ensure_ascii=False)
        rows.append(row)
    return frame.merge(pd.DataFrame(rows), on="gene_id", how="left", validate="one_to_one")
