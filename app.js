/**
 * app.js
 * Core logic for validating the required dataset files (training, testing, validation)
 * based on their exact file names before proceeding to the next step (e.g., model training).
 */

// --- Configuration: Expected File Names (FIXED) ---
const REQUIRED_FILES = {
    training: 'train.csv', // CORRECTED from 'training_data.csv'
    testing: 'test.csv',     // CORRECTED from 'testing_data.csv'
    validation: 'validation.csv' // CORRECTED from 'validation_data.csv'
};

// Map file input IDs to their expected name and status display ID
// This mapping still correctly references the updated REQUIRED_FILES object.
const fileMappings = [
    { id: 'trainingFile', expectedName: REQUIRED_FILES.training, statusId: 'trainingStatus' },
    { id: 'testingFile', expectedName: REQUIRED_FILES.testing, statusId: 'testingStatus' },
    { id: 'validationFile', expectedName: REQUIRED_FILES.validation, statusId: 'validationStatus' }
];

// Stores references to the File objects after they are loaded
const loadedFiles = {
    training: null,
    testing: null,
    validation: null
};

// --- Utility Functions ---

/**
 * Updates the status message for a specific file input.
 * @param {string} statusId The ID of the HTML element to update.
 * @param {string} message The message to display.
 * @param {string} type 'success', 'error', or 'info' to control styling.
 */
function updateFileStatus(statusId, message, type = 'info') {
    const statusElement = document.getElementById(statusId);
    if (!statusElement) return;

    // Reset classes
    statusElement.className = 'mt-2 text-sm';

    switch (type) {
        case 'success':
            statusElement.classList.add('text-green-600', 'font-semibold');
            break;
        case 'error':
            statusElement.classList.add('text-red-600', 'font-semibold');
            break;
        case 'info':
        default:
            statusElement.classList.add('text-gray-500');
            break;
    }
    statusElement.textContent = message;
}

/**
 * Updates the general status area at the bottom of the container and controls the button state.
 * @param {string} message The message to display.
 * @param {string} bgColor Tailwind background class for the container.
 * @param {string} textColor Tailwind text color class for the message.
 */
function updateGeneralStatus(message, bgColor = 'bg-gray-100', textColor = 'text-gray-600') {
    const generalStatus = document.getElementById('generalStatus');
    const processBtn = document.getElementById('processBtn');

    if (generalStatus) {
        generalStatus.className = `mt-6 p-4 rounded-lg text-center font-medium transition-all duration-300 ${bgColor} ${textColor}`;
        generalStatus.textContent = message;
    }
    
    // Control Button state based on file readiness and process status
    if (processBtn) {
        const allFilesReady = Object.values(loadedFiles).every(f => f !== null);
        // Button is disabled if files aren't ready OR if the app is currently processing (yellow status)
        processBtn.disabled = !allFilesReady || (bgColor === 'bg-yellow-100'); 
    }
}

/**
 * Handles the change event for any file input element.
 * @param {Event} event The change event.
 */
function handleFileChange(event) {
    const input = event.target;
    const file = input.files[0];

    // Find the corresponding mapping object
    const mapping = fileMappings.find(m => m.id === input.id);
    if (!file || !mapping) {
        // Clear status if file was removed
        updateFileStatus(mapping.statusId, 'File not selected.', 'error');
        loadedFiles[mapping.id.replace('File', '')] = null;
    } else if (file.name === mapping.expectedName) {
        // Success: Store the File object and update status
        updateFileStatus(mapping.statusId, `✅ Loaded: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`, 'success');
        // Store the file reference using its type (training, testing, validation)
        const fileType = mapping.id.replace('File', '');
        loadedFiles[fileType] = file;
    } else {
        // Error: File name mismatch
        updateFileStatus(
            mapping.statusId,
            `❌ Name mismatch. Uploaded: "${file.name}". Expected: "${mapping.expectedName}".`,
            'error'
        );
        // Clear the input and stored reference
        input.value = ''; 
        loadedFiles[mapping.id.replace('File', '')] = null;
    }

    // Check overall readiness and update general status and button state
    const allFilesReady = Object.values(loadedFiles).every(f => f !== null);
    if (allFilesReady) {
        updateGeneralStatus("All files successfully loaded. Ready to validate and prepare the model.", 'bg-indigo-100', 'text-indigo-800');
    } else {
        updateGeneralStatus("Please ensure all three files are selected and named correctly.");
    }
}

/**
 * Main function triggered by the "Validate & Prepare Model" button.
 * This simulates the successful transition to the next ML stage.
 */
function processFiles() {
    // Only run if the button isn't disabled (meaning files are ready)
    if (document.getElementById('processBtn').disabled) return;

    updateGeneralStatus("Validating file integrity and initiating model preparation...", 'bg-yellow-100', 'text-yellow-800');

    // Log the file details (where real processing would start)
    console.log("--- ML Stage Started ---");
    console.log("Training File:", loadedFiles.training.name, "Size:", loadedFiles.training.size);
    console.log("Testing File:", loadedFiles.testing.name, "Size:", loadedFiles.testing.size);
    console.log("Validation File:", loadedFiles.validation.name, "Size:", loadedFiles.validation.size);

    // Simulate an asynchronous processing delay
    setTimeout(() => {
        updateGeneralStatus("✅ Model Data Structure Confirmed! Ready for model training.", 'bg-green-100', 'text-green-800');
        // Next steps would involve using a library like Papaparse or the native FileReader API
        // to read the contents of loadedFiles.training, loadedFiles.testing, etc.
    }, 1500);

}

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    // 1. Attach the change handler to all file inputs
    fileMappings.forEach(mapping => {
        const inputElement = document.getElementById(mapping.id);
        if (inputElement) {
            inputElement.addEventListener('change', handleFileChange);
            // Initialize status text
            updateFileStatus(mapping.statusId, `Awaiting file "${mapping.expectedName}"...`, 'info');
        }
    });

    // 2. Attach the processFiles function to the button click event (FIX for ReferenceError)
    const processBtn = document.getElementById('processBtn');
    if (processBtn) {
        processBtn.addEventListener('click', processFiles);
    }
    
    // 3. Set the initial status and ensure the button is disabled
    updateGeneralStatus("Awaiting file selection...");
});
