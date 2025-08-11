# Global-INR-Trade-Settlement-proto-Type

Streamlit app (`main.py`) for INR trade settlement with UPI/NEFT/SRVA ingestion, AES-256-GCM encryption, Plotly analytics (risk, corridors, FX), and RBI compliance.  
Offers dashboard, analytics, markets, and security center.  
Scalable (10–50 to 1,000+ TPS), <1s ingestion / <3s insights.  
For banks, traders, regulators, and treasuries. *(280 characters)*

---

### Overview
This is a prototype for a real-time, RBI-aligned platform for INR-denominated cross-border trade settlement.  
Built as a single-file **Streamlit** application (`main.py`), it provides secure, transparent transaction processing and analytics for banks, exporters/importers, regulators, and treasuries.

---

### Features
- **Dual-rail ingestion:** UPI, NEFT, SRVA-INR, and SRVA-Non-INR transactions  
- **Security:** AES-256-GCM at rest, TLS 1.3 in transit, with key rotation  
- **Real-time analytics:** Plotly visualizations for risk scores, corridor stats, and FX volatility  
- **Regulatory compliance:** FEMA tagging, AFA alignment, and data minimization  
- **UI components:** Dashboard, transaction ingestion, advanced analytics, markets & FX, security center  
- **Performance:** 10–50 TPS (scalable to 1,000+), <1s ingestion, <3s insights  
- **Auto-generation:** Simulates real-time transaction data for testing  

---

### Installation

**Clone the repository**
```bash
git clone https://github.com/your-username/Global-INR-Trade-Settlement-proto-Type.git
cd Global-INR-Trade-Settlement-proto-Type
