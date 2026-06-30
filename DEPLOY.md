# Deployment Guide: Render (free tier)

These steps deploy this Flask API as a live, public URL — the piece that turns "I built a model" into "I deployed a model."

## 1. Push this project to GitHub
```bash
cd churn-api
git init
git add .
git commit -m "Churn prediction API with Docker and monitoring"
```
Create a new repo on GitHub (e.g. `churn-prediction-api`), then:
```bash
git remote add origin https://github.com/<your-username>/churn-prediction-api.git
git branch -M main
git push -u origin main
```

## 2. Create a Render account
Go to https://render.com and sign up (free, can use GitHub login).

## 3. Create a new Web Service
- Click **New +** → **Web Service**
- Connect your GitHub account and select the `churn-prediction-api` repo
- Render will detect the `Dockerfile` automatically — choose **Docker** as the environment
- Name: `churn-prediction-api` (this becomes part of your public URL)
- Instance type: **Free**
- Click **Create Web Service**

## 4. Wait for the build
Render will build the Docker image and deploy it. This takes a few minutes. You'll get a live URL like:
```
https://churn-prediction-api.onrender.com
```

## 5. Test the live deployment
```bash
curl https://churn-prediction-api.onrender.com/health

curl -X POST https://churn-prediction-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 2,
    "monthly_charges": 95.5,
    "total_charges": 191.0,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "tech_support": "No",
    "payment_method": "Electronic check"
  }'
```

## Notes on the free tier
- Render's free web services "spin down" after ~15 minutes of inactivity and take ~30-60 seconds to wake up on the next request. This is normal and fine for a portfolio project — just mention it if you're demoing live in an interview ("first request may take a moment to spin up").
- The `prediction_log.csv` file resets whenever the service restarts/redeploys on the free tier, since there's no persistent disk. That's expected — for a real production system you'd point logging at a managed database or logging service instead of a local file, which is worth mentioning if asked about this limitation in an interview (shows you understand the gap).

## Alternative: Hugging Face Spaces
If you'd rather host on Hugging Face Spaces (good for an ML-specific audience), the process is similar: create a Space, choose "Docker" as the SDK, and push this same repo content there instead. Worth doing as a second deployment target if you have time, since it signals familiarity with ML-specific platforms.
