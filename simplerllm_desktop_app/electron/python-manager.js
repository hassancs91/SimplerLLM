const { spawn } = require('child_process');
const path = require('path');
const { app } = require('electron');
const http = require('http');

class PythonManager {
    constructor() {
        this.process = null;
        this.port = 5123;
        this.isRunning = false;
    }

    getPythonPath() {
        if (app.isPackaged) {
            // Production: use bundled Python
            const resourcesPath = process.resourcesPath;
            if (process.platform === 'win32') {
                return path.join(resourcesPath, 'python', 'python.exe');
            } else {
                return path.join(resourcesPath, 'python', 'bin', 'python3');
            }
        } else {
            // Development: use system Python
            return process.platform === 'win32' ? 'python' : 'python3';
        }
    }

    getBackendPath() {
        if (app.isPackaged) {
            // Backend is unpacked from asar to app.asar.unpacked
            return path.join(process.resourcesPath, 'app.asar.unpacked', 'backend', 'app.py');
        } else {
            return path.join(__dirname, '..', 'backend', 'app.py');
        }
    }

    async start() {
        return new Promise((resolve, reject) => {
            const pythonPath = this.getPythonPath();
            const backendPath = this.getBackendPath();

            console.log(`Starting Python backend...`);
            console.log(`Python path: ${pythonPath}`);
            console.log(`Backend path: ${backendPath}`);

            const backendDir = path.dirname(backendPath);
            const env = {
                ...process.env,
                FLASK_PORT: this.port.toString(),
                PYTHONUNBUFFERED: '1',
                PYTHONPATH: backendDir
            };

            this.process = spawn(pythonPath, [backendPath], {
                env,
                cwd: path.dirname(backendPath),
                stdio: ['pipe', 'pipe', 'pipe']
            });

            this.process.stdout.on('data', (data) => {
                console.log(`[Python] ${data.toString().trim()}`);
            });

            this.process.stderr.on('data', (data) => {
                console.error(`[Python Error] ${data.toString().trim()}`);
            });

            this.process.on('error', (error) => {
                console.error('Failed to start Python process:', error);
                this.isRunning = false;
                reject(error);
            });

            this.process.on('close', (code) => {
                console.log(`Python process exited with code ${code}`);
                this.isRunning = false;
            });

            // Give the server a moment to start
            setTimeout(() => {
                this.isRunning = true;
                resolve();
            }, 2000);
        });
    }

    async waitForHealth(timeout = 30000) {
        const startTime = Date.now();

        while (Date.now() - startTime < timeout) {
            try {
                const isHealthy = await this.checkHealth();
                if (isHealthy) {
                    console.log('Backend health check passed');
                    return true;
                }
            } catch (error) {
                // Ignore errors, keep trying
            }
            await this.sleep(500);
        }

        console.error('Backend health check timed out');
        return false;
    }

    checkHealth() {
        return new Promise((resolve) => {
            const options = {
                hostname: 'localhost',
                port: this.port,
                path: '/api/health',
                method: 'GET',
                timeout: 2000
            };

            const req = http.request(options, (res) => {
                resolve(res.statusCode === 200);
            });

            req.on('error', () => resolve(false));
            req.on('timeout', () => {
                req.destroy();
                resolve(false);
            });

            req.end();
        });
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    stop() {
        if (this.process) {
            console.log('Stopping Python backend...');

            if (process.platform === 'win32') {
                // On Windows, we need to kill the process tree
                spawn('taskkill', ['/pid', this.process.pid, '/f', '/t']);
            } else {
                this.process.kill('SIGTERM');
            }

            this.process = null;
            this.isRunning = false;
        }
    }

    async restart() {
        this.stop();
        await this.sleep(1000);
        await this.start();
        return this.waitForHealth();
    }
}

module.exports = { PythonManager };
