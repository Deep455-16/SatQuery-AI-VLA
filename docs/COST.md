# SatQuery AI — Cost Documentation

**Last verified:** 2026-09-05  
**Context:** ISRO/SAC SIH Problem Statement 26167 Demo Deployment

## Cost Summary: ZERO for SIH Demonstration

All services used are on genuinely free tiers sufficient for the SIH prototype.

---

## 1. Google Gemini API — Free Tier

**Model:** `gemini-2.5-flash`  
**Source:** https://ai.google.dev/pricing (verified 2026-09-05)

| Metric | Free Tier |
|--------|-----------|
| Cost | $0 |
| Requests per minute | 15 RPM |
| Requests per day | 1,500 RPD |
| Tokens per minute | 1,000,000 TPM |
| Billing required | No |
| API key | Free from https://aistudio.google.com/apikey |

**SIH Demo Assessment:** ✅ Sufficient  
An SIH demonstration with judges asking questions interactively will use at most
5–20 requests during a session. Well within the 15 RPM / 1,500 RPD limits.

**Limitations of Free Tier:**
- Content may be used to improve Google products (see ToS)
- No SLA guarantee
- Rate-limited to 15 RPM
- Not suitable for production serving at scale

---

## 2. MongoDB Atlas — M0 Free Cluster

**Source:** https://www.mongodb.com/pricing (verified)

| Metric | M0 Free |
|--------|---------|
| Cost | $0/month permanently |
| Storage | 512 MB |
| RAM | Shared |
| Connections | 500 max |
| Credit card | Not required |
| Expiry | Never expires |

**SIH Demo Assessment:** ✅ Sufficient  
Analysis history, session metadata, and image records for an SIH demo
will consume well under 100MB. 512MB is more than adequate.

**Limitations:**
- Shared cluster (variable performance)
- No advanced analytics
- Limited to 5 projects per organization on free tier

---

## 3. Local Computation — No Cost

All satellite image preprocessing runs locally on CPU:
- Rasterio (GeoTIFF parsing) — free/open-source
- OpenCV (change detection) — free/open-source
- NumPy, scikit-image — free/open-source
- PIL/Pillow — free/open-source

**Hardware requirement:** Standard laptop/desktop CPU. No GPU required.

---

## 4. No Paid Services Used

The following are explicitly **NOT** used:
- OpenAI API (paid)
- Anthropic Claude API (paid)
- Azure Cognitive Services (paid)
- AWS Rekognition (paid)
- Google Cloud Vision API (paid)
- Hugging Face Inference API (paid)
- Any paid GPU cloud service
- Any paid vector database
- Any paid geospatial API

---

## 5. Optional Specialist Models (Research/Full Mode Only)

If a researcher activates specialist models (TerraQ-VL, GeoChat, VisTA, DS_UNet),
these require:
- Local GPU with sufficient VRAM (typically 8–24GB)
- Model weights downloaded from HuggingFace or GitHub (free but large)
- No additional monetary cost (open-source weights)

These are **not required** for the SIH free-cloud-demo mode.

---

## Conclusion

**Total monetary cost for SIH demonstration: $0.00**

The system operates entirely on free-tier cloud services supplemented by
local open-source image processing. No credit card or paid subscription
is required to demonstrate SatQuery AI to SIH judges.
