#!/usr/bin/env bash
set -euo pipefail

python scripts/run_train_agent.py --algo ppo --experiment-name overnight_ppo
python scripts/run_train_agent.py --algo sac --experiment-name overnight_sac
