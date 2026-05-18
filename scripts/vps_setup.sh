#!/usr/bin/env bash
# One-time VPS setup script (Ubuntu/Debian)
# Usage: bash scripts/vps_setup.sh <github_repo_url>
# Example: bash scripts/vps_setup.sh https://github.com/yourname/weather-forecast-ml.git

set -euo pipefail

REPO_URL="${1:-}"
APP_DIR="/opt/weather-forecast-ml"

if [[ -z "$REPO_URL" ]]; then
    echo "Usage: bash scripts/vps_setup.sh <github_repo_url>"
    exit 1
fi

echo "=== [1/6] Installing system packages ==="
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git curl

# confirm Python >= 3.10
PYTHON_VERSION=$(python3 -c "import sys; print(sys.version_info.minor)")
if [[ "$PYTHON_VERSION" -lt 10 ]]; then
    echo "Python 3.10+ required. Installing 3.11..."
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get install -y python3.11 python3.11-venv
    PYTHON_BIN="python3.11"
else
    PYTHON_BIN="python3"
fi

echo "=== [2/6] Cloning repository ==="
if [[ -d "$APP_DIR" ]]; then
    echo "Directory $APP_DIR already exists — pulling latest"
    git -C "$APP_DIR" pull
else
    git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

echo "=== [3/6] Creating virtual environment ==="
$PYTHON_BIN -m venv .venv
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -r requirements.txt --quiet

echo "=== [4/6] Creating .env ==="
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo ""
    echo "  Edit .env before continuing:"
    echo "    nano $APP_DIR/.env"
    echo ""
    echo "  Required fields:"
    echo "    TELEGRAM_BOT_TOKEN=..."
    echo "    TELEGRAM_CHAT_ID=..."
    echo ""
    read -rp "  Press Enter after editing .env ..."
fi

echo "=== [5/6] Downloading historical data (~10 min) ==="
mkdir -p data/raw data/processed models logs
.venv/bin/python src/download_history.py

echo "=== [5/6] Training models ==="
.venv/bin/python src/train.py

echo "=== [6/6] Setting up cron jobs ==="
PYTHON_PATH="$APP_DIR/.venv/bin/python"
CRON_DAILY="0 7 * * * cd $APP_DIR && $PYTHON_PATH src/telegram_bot.py >> $APP_DIR/logs/telegram.log 2>&1"
CRON_HOURLY="0 * * * * cd $APP_DIR && $PYTHON_PATH src/fetch_live.py >> $APP_DIR/logs/fetch.log 2>&1"

# add cron jobs without duplicating
( crontab -l 2>/dev/null | grep -v "telegram_bot\|fetch_live" ; \
  echo "$CRON_DAILY" ; echo "$CRON_HOURLY" ) | crontab -

echo ""
echo "====================================================="
echo "  Setup complete!"
echo "  Daily Telegram forecast: 07:00 every day"
echo "  Hourly live data fetch : every hour"
echo ""
echo "  Test Telegram now:"
echo "    cd $APP_DIR && .venv/bin/python src/telegram_bot.py"
echo "====================================================="
