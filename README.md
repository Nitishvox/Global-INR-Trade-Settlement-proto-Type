# Global-INR-Trade-Settlement-proto-Type

Streamlit app (`main.py`) for INR trade settlement with UPI/NEFT/SRVA ingestion, AES-256-GCM encryption, Plotly analytics (risk, corridors, FX), and RBI compliance.  
Offers dashboard, analytics, markets, and security center.  
Scalable (10–50 to 1,000+ TPS), <1s ingestion / <3s insights.  
For banks, traders, regulators, and treasuries. *(280 characters)*

---

## Prototype Disclaimer

This repository contains **prototype software** — not finished, production-ready code.  
It is intended for experimentation, architecture validation, and demonstration only.

If you plan to deploy in production, perform:
- Security audits  
- Compliance checks  
- Integration testing  
- Performance tuning  

---

## Overview

A real-time, RBI-aligned platform prototype for INR-denominated cross-border trade settlement.

Built as a single-file Streamlit application (`main.py`), it enables secure, transparent transaction processing and analytics for:
- Banks  
- Exporters / Importers  
- Regulators  
- Treasuries  

---

## Features

- Dual-rail ingestion: UPI, NEFT, RTGS, SRVA-INR, SRVA-Non-INR, SWIFT  
- Security: AES-256-GCM encryption (at rest), TLS 1.3 (in transit), key rotation  
- Analytics: Plotly visualizations — risk heatmaps, volume treemaps, FX volatility  
- Performance Metrics: TPS, ingestion latency, analytics latency, SRVA utilization, transaction count  
- Compliance: FEMA tagging, AFA alignment, RBI data minimization  
- UI Components: Dashboard, Ingestion, Analytics, Markets & FX, Security Center, Settings  
- Auto-generation: Simulates 10, 1,000, or 10,000 transactions  
- Data Pipeline: Excel upload for batch processing and risk prediction  

---
## Hosted App

Access the prototype directly via the hosted Streamlit app:  
[Global-INR-Trade-Settlement Viewer](https://global-inr-trade-settlement-proto-type-sjdhxpmvp3rrk38a2oxyru.streamlit.app/)


## Installation

### Option 1 — Local Setup

```bash
# Clone the repository
git clone https://github.com/your-username/Global-INR-Trade-Settlement-proto-Type.git
cd Global-INR-Trade-Settlement-proto-Type

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run main.py
