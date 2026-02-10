/**
 * Setup Portable Python for Packaging
 *
 * This script downloads and sets up an embedded Python distribution
 * with all required dependencies for the SimplerLLM Playground.
 * Supports Windows, macOS, and Linux.
 */

const https = require('https');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Configuration
const PYTHON_VERSION = '3.11.7';
const STANDALONE_VERSION = '20240107';

// Platform-specific Python URLs (using python-build-standalone for cross-platform)
const PYTHON_URLS = {
    win32: `https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-embed-amd64.zip`,
    darwin: `https://github.com/indygreg/python-build-standalone/releases/download/${STANDALONE_VERSION}/cpython-${PYTHON_VERSION}+${STANDALONE_VERSION}-aarch64-apple-darwin-install_only.tar.gz`,
    darwinX64: `https://github.com/indygreg/python-build-standalone/releases/download/${STANDALONE_VERSION}/cpython-${PYTHON_VERSION}+${STANDALONE_VERSION}-x86_64-apple-darwin-install_only.tar.gz`,
    linux: `https://github.com/indygreg/python-build-standalone/releases/download/${STANDALONE_VERSION}/cpython-${PYTHON_VERSION}+${STANDALONE_VERSION}-x86_64-unknown-linux-gnu-install_only.tar.gz`
};

const GET_PIP_URL = 'https://bootstrap.pypa.io/get-pip.py';

const ROOT_DIR = path.join(__dirname, '..');
const PYTHON_DIR = path.join(ROOT_DIR, 'python');
const BACKEND_DIR = path.join(ROOT_DIR, 'backend');

// Platform detection
const PLATFORM = process.platform;
const ARCH = process.arch;
const IS_WINDOWS = PLATFORM === 'win32';
const IS_MAC = PLATFORM === 'darwin';
const IS_LINUX = PLATFORM === 'linux';

// Get Python executable path based on platform
function getPythonExe() {
    if (IS_WINDOWS) {
        return path.join(PYTHON_DIR, 'python.exe');
    } else {
        // python-build-standalone extracts to python/bin/python3
        return path.join(PYTHON_DIR, 'bin', 'python3');
    }
}

// Get the correct download URL for current platform
function getPythonUrl() {
    if (IS_WINDOWS) {
        return PYTHON_URLS.win32;
    } else if (IS_MAC) {
        return ARCH === 'arm64' ? PYTHON_URLS.darwin : PYTHON_URLS.darwinX64;
    } else {
        return PYTHON_URLS.linux;
    }
}

/**
 * Download a file from URL
 */
function downloadFile(url, dest) {
    return new Promise((resolve, reject) => {
        console.log(`Downloading: ${url}`);

        const file = fs.createWriteStream(dest);

        https.get(url, (response) => {
            if (response.statusCode === 302 || response.statusCode === 301) {
                // Handle redirect
                downloadFile(response.headers.location, dest)
                    .then(resolve)
                    .catch(reject);
                return;
            }

            if (response.statusCode !== 200) {
                reject(new Error(`HTTP ${response.statusCode}`));
                return;
            }

            const totalSize = parseInt(response.headers['content-length'], 10);
            let downloadedSize = 0;

            response.on('data', (chunk) => {
                downloadedSize += chunk.length;
                if (totalSize) {
                    const percent = ((downloadedSize / totalSize) * 100).toFixed(1);
                    process.stdout.write(`\rProgress: ${percent}%`);
                }
            });

            response.pipe(file);

            file.on('finish', () => {
                file.close();
                console.log('\nDownload complete!');
                resolve();
            });
        }).on('error', (err) => {
            fs.unlink(dest, () => {});
            reject(err);
        });
    });
}

/**
 * Extract archive (ZIP for Windows, tar.gz for macOS/Linux)
 */
function extractArchive(archivePath, destDir) {
    console.log(`Extracting to: ${destDir}`);

    if (IS_WINDOWS) {
        const AdmZip = require('adm-zip');
        const zip = new AdmZip(archivePath);
        zip.extractAllTo(destDir, true);
    } else {
        // Use tar for macOS/Linux
        // python-build-standalone extracts to a 'python' subfolder, so we extract to parent
        execSync(`tar -xzf "${archivePath}" -C "${path.dirname(destDir)}"`, {
            stdio: 'inherit'
        });
    }

    console.log('Extraction complete!');
}

/**
 * Enable pip in embedded Python (Windows only)
 */
function enablePip() {
    if (!IS_WINDOWS) {
        console.log('Skipping pip enable (not needed for standalone Python)');
        return;
    }

    console.log('Enabling pip...');

    // Find the ._pth file (e.g., python311._pth)
    const files = fs.readdirSync(PYTHON_DIR);
    const pthFile = files.find(f => f.endsWith('._pth'));

    if (!pthFile) {
        throw new Error('Could not find ._pth file');
    }

    const pthPath = path.join(PYTHON_DIR, pthFile);
    let content = fs.readFileSync(pthPath, 'utf-8');

    // Uncomment 'import site'
    content = content.replace('#import site', 'import site');

    // Add Lib/site-packages to path
    if (!content.includes('Lib/site-packages')) {
        content += '\nLib/site-packages\n';
    }

    fs.writeFileSync(pthPath, content);
    console.log('pip enabled!');
}

/**
 * Install pip (Windows embedded Python only - standalone already has pip)
 */
async function installPip() {
    const pythonExe = getPythonExe();

    // Check if pip already exists (standalone Python has it)
    try {
        execSync(`"${pythonExe}" -m pip --version`, { stdio: 'pipe' });
        console.log('pip already available, skipping installation');
        return;
    } catch (e) {
        // pip not found, need to install
    }

    console.log('Installing pip...');

    const getPipPath = path.join(PYTHON_DIR, 'get-pip.py');
    await downloadFile(GET_PIP_URL, getPipPath);

    execSync(`"${pythonExe}" "${getPipPath}"`, {
        cwd: PYTHON_DIR,
        stdio: 'inherit'
    });

    // Clean up
    fs.unlinkSync(getPipPath);
    console.log('pip installed!');
}

/**
 * Install Python dependencies
 */
function installDependencies() {
    console.log('Installing dependencies...');

    const pythonExe = getPythonExe();
    const requirementsPath = path.join(BACKEND_DIR, 'requirements.txt');

    // Upgrade pip first
    execSync(`"${pythonExe}" -m pip install --upgrade pip`, {
        cwd: PYTHON_DIR,
        stdio: 'inherit'
    });

    // Install requirements
    execSync(`"${pythonExe}" -m pip install -r "${requirementsPath}"`, {
        cwd: PYTHON_DIR,
        stdio: 'inherit'
    });

    console.log('Dependencies installed!');
}

/**
 * Main setup function
 */
async function setup() {
    console.log('========================================');
    console.log('SimplerLLM Playground - Python Setup');
    console.log(`Platform: ${PLATFORM} (${ARCH})`);
    console.log('========================================\n');

    try {
        const pythonExe = getPythonExe();

        // Check if already set up
        if (fs.existsSync(pythonExe)) {
            console.log('Python already set up. Skipping download.');
            console.log('To reinstall, delete the python/ folder and run again.\n');

            // Just install/update dependencies
            installDependencies();
            return;
        }

        // Create python directory
        if (!fs.existsSync(PYTHON_DIR)) {
            fs.mkdirSync(PYTHON_DIR, { recursive: true });
        }

        // Download Python
        const archiveExt = IS_WINDOWS ? 'zip' : 'tar.gz';
        const archivePath = path.join(ROOT_DIR, `python-embed.${archiveExt}`);
        await downloadFile(getPythonUrl(), archivePath);

        // Extract Python
        extractArchive(archivePath, PYTHON_DIR);

        // Clean up archive
        fs.unlinkSync(archivePath);

        // Enable pip (Windows only)
        enablePip();

        // Install pip
        await installPip();

        // Install dependencies
        installDependencies();

        console.log('\n========================================');
        console.log('Setup complete!');
        console.log('========================================');
        const buildCmd = IS_WINDOWS ? 'npm run build:win' : (IS_MAC ? 'npm run build:mac' : 'npm run build:linux');
        console.log(`\nYou can now run: ${buildCmd}`);

    } catch (error) {
        console.error('\nSetup failed:', error.message);
        process.exit(1);
    }
}

// Run setup
setup();
