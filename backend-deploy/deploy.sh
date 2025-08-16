#!/bin/bash

echo "🚀 Deploying MyWardrobe to Railway..."

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Login to Railway
echo "🔐 Logging into Railway..."
railway login

# Initialize project if not already done
if [ ! -f "railway.json" ]; then
    echo "📁 Initializing Railway project..."
    railway init
fi

# Deploy
echo "🚂 Deploying to Railway..."
railway up

echo "✅ Deployment complete!"
echo "🌐 Your app should be available at the URL shown above"
echo "📊 Check Railway dashboard for logs and status"
