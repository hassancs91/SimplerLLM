const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods to the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
    // Window controls
    minimizeWindow: () => ipcRenderer.invoke('window:minimize'),
    maximizeWindow: () => ipcRenderer.invoke('window:maximize'),
    closeWindow: () => ipcRenderer.invoke('window:close'),

    // Backend communication
    getBackendPort: () => ipcRenderer.invoke('backend:getPort'),
    getBackendStatus: () => ipcRenderer.invoke('backend:getStatus'),
    restartBackend: () => ipcRenderer.invoke('backend:restart'),

    // Event listeners
    onBackendStatus: (callback) => {
        ipcRenderer.on('backend:status', (event, status) => callback(status));
    },

    // File system
    showItemInFolder: (filePath) => ipcRenderer.invoke('shell:showItemInFolder', filePath),

    // File dialog
    openFileDialog: (options) => ipcRenderer.invoke('dialog:openFile', options)
});

// Log when preload script is loaded
console.log('Preload script loaded');
