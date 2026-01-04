import json
import os

# Path to your settings file
config_file = "settings.json"

if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        data = json.load(f)

    # FORCE RESET SCALE
    data["ui_scale"] = 1.0

    with open(config_file, 'w') as f:
        json.dump(data, f, indent=2)

    print("✅ UI Scale reset to 1.0. You can now run the app.")
else:
    print("❌ settings.json not found.")