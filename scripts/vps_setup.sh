#!/usr/bin/env bash
# One-time VPS setup script (Ubuntu/Debian)
# Run from inside the cloned repo directory:
#   cd /path/to/weather-forecast-ml
#   bash scripts/vps_setup.sh

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

echo "=== Working directory: $APP_DIR ==="

echo ""
echo "=== [1/5] Installing system packages ==="
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git

PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VER"
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [[ "$PYTHON_MINOR" -lt 10 ]]; then
    echo "Installing Python 3.11 (current version is too old)..."
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get install -y python3.11 python3.11-venv
    PYTHON_BIN="python3.11"
else
    PYTHON_BIN="python3"
fi

echo ""
echo "=== [2/5] Creating virtual environment ==="
$PYTHON_BIN -m venv .venv
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -r requirements.txt --quiet
echo "Dependencies installed."

echo ""
echo "=== [3/5] Creating .env ==="
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo ""
    echo "  .env created from template. Edit it now:"
    echo "    nano $APP_DIR/.env"
    echo ""
    echo "  Required fields:"
    echo "    TELEGRAM_BOT_TOKEN=..."
    echo "    TELEGRAM_CHAT_ID=..."
    echo ""
    read -rp "  Press Enter after saving .env ..."
else
    echo "  .env already exists — skipping."
fi

echo ""
echo "=== [4/5] Downloading data + training models ==="
mkdir -p data/raw data/processed models logs
echo "  Downloading 20 years of weather data (~10 min)..."
.venv/bin/python src/download_history.py
echo "  Training XGBoost models..."
.venv/bin/python src/train.py

echo ""
echo "=== [5/5] Setting up cron jobs ==="
PYTHON_PATH="$APP_DIR/.venv/bin/python"
CRON_DAILY="0 7 * * * cd $APP_DIR && $PYTHON_PATH src/telegram_bot.py >> $APP_DIR/logs/telegram.log 2>&1"
CRON_HOURLY="0 * * * * cd $APP_DIR && $PYTHON_PATH src/fetch_live.py >> $APP_DIR/logs/fetch.log 2>&1"

( crontab -l 2>/dev/null | grep -v "telegram_bot\|fetch_live" ; \
  echo "$CRON_DAILY" ; echo "$CRON_HOURLY" ) | crontab -

echo ""
echo "====================================================="
echo "  Setup complete!"
echo "  Daily Telegram forecast : 07:00 every day"
echo "  Hourly live data fetch  : every hour"
echo ""
echo "  Test Telegram now:"
echo "    cd $APP_DIR && .venv/bin/python src/telegram_bot.py"
echo "====================================================="
