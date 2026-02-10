# SimplerLLM Playground - Build Guide

## Quick Commands

```bash
# Test the app locally
npm start

# Build for Windows
npm run build:win

# Build for macOS
npm run build:mac

# Build for Linux
npm run build:linux

# Build for all platforms
npm run build
```

## How to Release a New Version

### Step 1: Update Version Numbers

Update the version in these files:

1. **package.json** - Line 3:
   ```json
   "version": "1.0.0"  →  "version": "1.1.0"
   ```

2. **renderer/js/branding.json** - Line 3:
   ```json
   "version": "1.0.0"  →  "version": "1.1.0"
   ```

### Step 2: Build the App

```bash
# Navigate to the app folder
cd simplerllm_desktop_app

# Install dependencies (if not done)
npm install

# Build for your platform
npm run build:win     # Windows
npm run build:mac     # macOS
npm run build:linux   # Linux
```

### Step 3: Find Your Build

After building, find your installer in the `dist/` folder:

| Platform | Files |
|----------|-------|
| Windows | `SimplerLLM Playground Setup 1.x.x.exe` (installer), `SimplerLLM Playground 1.x.x.exe` (portable) |
| macOS | `SimplerLLM Playground-1.x.x.dmg`, `SimplerLLM Playground-1.x.x-mac.zip` |
| Linux | `SimplerLLM Playground-1.x.x.AppImage`, `simplerllm-playground_1.x.x_amd64.deb` |

---

## Version Numbering Guide

Use semantic versioning: `MAJOR.MINOR.PATCH`

| Change Type | Example | When to Use |
|-------------|---------|-------------|
| MAJOR (1.0.0 → 2.0.0) | Breaking changes | Major redesign, incompatible changes |
| MINOR (1.0.0 → 1.1.0) | New features | Added new tool, new functionality |
| PATCH (1.0.0 → 1.0.1) | Bug fixes | Fixed bugs, small improvements |

---

## Updating Branding/Metadata

| What to Change | File |
|----------------|------|
| App name, copyright, social links | `renderer/js/branding.json` |
| Package name, author, description | `package.json` |
| App ID, installer settings | `electron-builder.json` |
| App icons | `renderer/assets/icons/` |

---

## Troubleshooting

**Build fails with icon error:**
- Ensure all 3 icon files exist in `renderer/assets/icons/`:
  - `icon.ico` (Windows)
  - `icon.icns` (macOS)
  - `icon.png` (Linux)

**Python backend not bundled:**
- Run `npm run setup-python` before building
- Or use `npm run build` which runs setup automatically
