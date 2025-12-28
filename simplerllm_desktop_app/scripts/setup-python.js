/**
 * Setup Portable Python for Packaging
 *
 * This script downloads and sets up an embedded Python distribution
 * with all required dependencies for the SimplerLLM Playground.
 */

const https = require('https');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Configuration
const PYTHON_VERSION = '3.11.7';
const PYTHON_URL = `https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-embed-amd64.zip`;
const GET_PIP_URL = 'https://bootstrap.pypa.io/get-pip.py';

const ROOT_DIR = path.join(__dirname, '..');
const PYTHON_DIR = path.join(ROOT_DIR, 'python');
const BACKEND_DIR = path.join(ROOT_DIR, 'backend');

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
 * Extract ZIP file (requires adm-zip)
 */
function extractZip(zipPath, destDir) {
    console.log(`Extracting to: ${destDir}`);

    const AdmZip = require('adm-zip');
    const zip = new AdmZip(zipPath);
    zip.extractAllTo(destDir, true);

    console.log('Extraction complete!');
}

/**
 * Enable pip in embedded Python
 */
function enablePip() {
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
 * Install pip
 */
async function installPip() {
    console.log('Installing pip...');

    const getPipPath = path.join(PYTHON_DIR, 'get-pip.py');
    await downloadFile(GET_PIP_URL, getPipPath);

    const pythonExe = path.join(PYTHON_DIR, 'python.exe');

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

    const pythonExe = path.join(PYTHON_DIR, 'python.exe');
    const pipExe = path.join(PYTHON_DIR, 'Scripts', 'pip.exe');
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
    console.log('========================================\n');

    try {
        // Check if already set up
        if (fs.existsSync(path.join(PYTHON_DIR, 'python.exe'))) {
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
        const zipPath = path.join(ROOT_DIR, 'python-embed.zip');
        await downloadFile(PYTHON_URL, zipPath);

        // Extract Python
        extractZip(zipPath, PYTHON_DIR);

        // Clean up zip
        fs.unlinkSync(zipPath);

        // Enable pip
        enablePip();

        // Install pip
        await installPip();

        // Install dependencies
        installDependencies();

        console.log('\n========================================');
        console.log('Setup complete!');
        console.log('========================================');
        console.log('\nYou can now run: npm run build:win');

    } catch (error) {
        console.error('\nSetup failed:', error.message);
        process.exit(1);
    }
}

// Run setup
setup();
