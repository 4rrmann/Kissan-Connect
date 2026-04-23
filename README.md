# 🌱 Kissan Connect  -AI Agriculture Platform

An end-to-end Flask web application that helps Indian farmers with:
- **Crop Recommendation** (Random Forest, 98.4% accuracy)
- **Crop Disease Detection** (CNN / Keras)
- **Yield Prediction** (Gradient Boosting)
- **Kisaan Bot** (Hinglish chatbot, TF-IDF + Groq LLM)

---

## 🚀 Quick Start

```bash
cd kissan-connect
pip install -r requirements.txt

cp .env.example .env
# Add your GROQ_API_KEY

python train_models.py
python app.py