const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { PythonManager } = require('./python-manager');

let mainWindow;
let pythonManager;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        minWidth: 900,
        minHeight: 600,
        icon: path.join(__dirname, '..', 'renderer', 'assets', 'icons', 'icon.ico'),
        frame: false,
        titleBarStyle: 'hidden',
        backgroundColor: '#1A1A2E',
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false
        }
    });

    mainWindow.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));

    // Open DevTools in development
    if (!app.isPackaged) {
        mainWindow.webContents.openDevTools();
    }

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

async function startApp() {
    // Start Python/Flask backend
    pythonManager = new PythonManager();

    try {
        await pythonManager.start();
        console.log('Python backend started successfully');

        // Wait for backend to be healthy
        const isHealthy = await pythonManager.waitForHealth();
        if (isHealthy) {
            console.log('Backend is healthy, creating window...');
            createWindow();
        } else {
            console.error('Backend health check failed');
            createWindow(); // Still create window to show error state
        }
    } catch (error) {
        console.error('Failed to start Python backend:', error);
        createWindow(); // Create window anyway to show error
    }
}

// IPC Handlers for window controls
ipcMain.handle('window:minimize', () => {
    if (mainWindow) mainWindow.minimize();
});

ipcMain.handle('window:maximize', () => {
    if (mainWindow) {
        if (mainWindow.isMaximized()) {
            mainWindow.unmaximize();
        } else {
            mainWindow.maximize();
        }
    }
});

ipcMain.handle('window:close', () => {
    if (mainWindow) mainWindow.close();
});

ipcMain.handle('backend:getPort', () => {
    return pythonManager ? pythonManager.port : 5123;
});

ipcMain.handle('backend:getStatus', () => {
    return pythonManager ? pythonManager.isRunning : false;
});

ipcMain.handle('backend:restart', async () => {
    if (pythonManager) {
        await pythonManager.restart();
        return true;
    }
    return false;
});

// App lifecycle
app.whenReady().then(startApp);

app.on('window-all-closed', () => {
    // Stop Python backend
    if (pythonManager) {
        pythonManager.stop();
    }

    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});

app.on('before-quit', () => {
    if (pythonManager) {
        pythonManager.stop();
    }
});
