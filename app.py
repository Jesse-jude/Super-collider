"""
AI Supercollider — Flask Server
Serves the UI at / and the REST API at /api/
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import traceback

from physics import (
    EventGenerator, AnomalyDetector,
    PARTICLES, get_cross_section
)

app = Flask(__name__)
CORS(app)

# Global engine instances (lightweight, fine for Render free tier)
_generators = {"physics": EventGenerator("physics"), "ai": EventGenerator("ai")}
_anomaly    = AnomalyDetector(window=100)


# ── UI ────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/api/collide", methods=["POST"])
def collide():
    """
    POST /api/collide
    {
      "beam_a":      "proton",
      "beam_b":      "antiproton",
      "energy_a_tev": 6.5,
      "energy_b_tev": 6.5,
      "angle_a_deg":  0,
      "angle_b_deg":  180,
      "mode":         "ai"   // "physics" | "ai"
    }
    """
    try:
        body   = request.get_json(force=True) or {}
        beam_a = body.get("beam_a",      "proton")
        beam_b = body.get("beam_b",      "antiproton")
        ea     = float(body.get("energy_a_tev", 6.5))
        eb     = float(body.get("energy_b_tev", 6.5))
        ang_a  = float(body.get("angle_a_deg",  0))
        ang_b  = float(body.get("angle_b_deg",  180))
        mode   = body.get("mode", "physics")

        if beam_a not in PARTICLES:
            return jsonify(error=f"Unknown particle: {beam_a}"), 400
        if beam_b not in PARTICLES:
            return jsonify(error=f"Unknown particle: {beam_b}"), 400
        if not (0.1 <= ea <= 14) or not (0.1 <= eb <= 14):
            return jsonify(error="Energy must be 0.1–14 TeV"), 400

        gen  = _generators.get(mode, _generators["physics"])
        pa   = gen.make_beam(beam_a, ea, ang_a)
        pb   = gen.make_beam(beam_b, eb, ang_b)
        ev   = gen.run_event(pa, pb)

        _anomaly.update(ev)
        ev["anomaly_detected"] = _anomaly.is_anomaly(ev)
        ev["anomaly_sigma"]    = round(_anomaly.score(ev), 2)

        return jsonify(ev)

    except Exception:
        return jsonify(error=traceback.format_exc()), 500


@app.route("/api/batch", methods=["POST"])
def batch():
    """
    POST /api/batch
    { "beam_a":"proton","beam_b":"proton","energy_tev":6.5,"n":500,"mode":"ai" }
    Returns aggregate statistics over n events.
    """
    try:
        body  = request.get_json(force=True) or {}
        n     = min(int(body.get("n", 500)), 5000)  # cap at 5k for free tier
        mode  = body.get("mode", "physics")
        gen   = _generators.get(mode, _generators["physics"])
        stats = gen.run_batch(
            body.get("beam_a", "proton"),
            body.get("beam_b", "proton"),
            float(body.get("energy_tev", 6.5)),
            n=n,
        )
        return jsonify(stats)

    except Exception:
        return jsonify(error=traceback.format_exc()), 500


@app.route("/api/particles", methods=["GET"])
def particles():
    """GET /api/particles — returns the full particle database"""
    return jsonify(PARTICLES)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(status="ok", version="1.0.0")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
