# AI Supercollider

Particle collision simulator with physics engine + AI prediction layer.
Hosted on Render — UI + REST API in one service.

---

## Project Structure

```
supercollider-app/
├── app.py            ← Flask server (UI + API)
├── physics.py        ← Physics & AI engine
├── requirements.txt  ← Python dependencies
├── render.yaml       ← Render deploy config
└── templates/
    └── index.html    ← Full interactive UI
```

---

## Deploy to Render (5 minutes)

### Step 1 — Push to GitHub

```bash
cd supercollider-app
git init
git add .
git commit -m "Initial deploy"
# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/ai-supercollider.git
git push -u origin main
```

### Step 2 — Deploy on Render

1. Go to https://render.com and sign in (free account)
2. Click **New → Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — just click **Deploy**
5. Wait ~2 minutes for build

Your app will be live at: `https://ai-supercollider.onrender.com`

---

## API Reference

### POST /api/collide
Run a single collision event.

```bash
curl -X POST https://YOUR-APP.onrender.com/api/collide \
  -H "Content-Type: application/json" \
  -d '{
    "beam_a": "proton",
    "beam_b": "antiproton",
    "energy_a_tev": 6.5,
    "energy_b_tev": 6.5,
    "angle_a_deg": 0,
    "angle_b_deg": 180,
    "mode": "ai"
  }'
```

**Response:**
```json
{
  "id": 1,
  "sqrt_s_tev": 13.0,
  "beam_a": "proton",
  "beam_b": "antiproton",
  "n_particles": 18,
  "mode": "ai",
  "confidence": 0.91,
  "anomaly_detected": false,
  "anomaly_sigma": 0.4,
  "particles": [
    {
      "name": "W_plus",
      "energy_tev": 1.23,
      "px": 0.44, "py": -0.81, "pz": 1.12,
      "charge": 1.0,
      "mass_gev": 80.4,
      "spin": 1
    },
    ...
  ]
}
```

### POST /api/batch
Run many events and get statistics.

```bash
curl -X POST https://YOUR-APP.onrender.com/api/batch \
  -H "Content-Type: application/json" \
  -d '{
    "beam_a": "proton",
    "beam_b": "proton",
    "energy_tev": 6.5,
    "n": 1000,
    "mode": "physics"
  }'
```

### GET /api/particles
Full particle database (charge, mass, spin, quantum numbers).

### GET /api/health
Health check — returns `{"status": "ok"}`.

---

## Run Locally

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

---

## Notes

- Render free tier spins down after 15 min inactivity — first request may take ~30s to wake up.
- Batch endpoint is capped at 5000 events per call on free tier.
- To upgrade performance: change `--workers 2` to `--workers 4` in `render.yaml`.
