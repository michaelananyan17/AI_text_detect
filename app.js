/**
 * AI Text Detection ML Pipeline Logic
 */

// --- CONFIGURATION & GLOBAL STATE ---
const REQUIRED_FILES = {
    training: 'train.csv', 
    testing: 'test.csv',
    validation: 'validation.csv'
};
const fileMappings = [
    { id: 'trainingFile', expectedName: REQUIRED_FILES.training, statusId: 'trainingStatus' },
    { id: 'testingFile', expectedName: REQUIRED_FILES.testing, statusId: 'testingStatus' },
    { id: 'validationFile', expectedName: REQUIRED_FILES.validation, statusId: 'validationStatus' }
];

// Note: I renamed 'processedData' to 'normalizedData' in the JS to better reflect its function, 
// which is cleaning the raw data before tensor conversion.
const loadedFiles = { training: null, testing: null, validation: null };
const rawParsedData = { training: null, testing: null, validation: null }; 
const normalizedData = { training: null, testing: null, validation: null }; 

const MAX_SEQUENCE_LENGTH = 50; // Fixed sequence length for padding
const EMBEDDING_DIM = 16;       // Fixed size for the embedding vector

let model = null;
let wordIndex = {}; // Vocabulary map: word -> index
let VOCAB_SIZE = 0; // Calculated size of vocabulary

// --- UTILITY FUNCTIONS ---

/** Updates status message for file inputs. */
function updateFileStatus(statusId, message, type = 'info') {
    const statusElement = document.getElementById(statusId);
    if (!statusElement) return;

    statusElement.className = 'mt-1 text-sm font-medium';

    switch (type) {
        case 'success':
            statusElement.classList.add('text-green-600');
            break;
        case 'error':
            statusElement.classList.add('text-red-600');
            break;
        case 'info':
        default:
            statusElement.classList.add('text-gray-500');
            break;
    }
    statusElement.textContent = message;
}

/** Updates the main status area and controls the button state. */
function updateGeneralStatus(message, bgColor = 'bg-gray-100', textColor = 'text-gray-600', enableProcess = false) {
    const generalStatus = document.getElementById('generalStatus');
    const processBtn = document.getElementById('processBtn');

    if (generalStatus) {
        generalStatus.className = `mt-4 p-3 rounded-xl text-center font-medium transition-all duration-300 ${bgColor} ${textColor}`;
        generalStatus.innerHTML = message;
    }
    
    if (processBtn) {
        // Only disable the button if not all files are loaded, unless a data parsing error occurred (bg-yellow-100 is for processing)
        const allFilesReady = Object.values(loadedFiles).every(f => f !== null);
        // Only enable if files are ready AND no active processing is happening (bg-yellow)
        processBtn.disabled = !allFilesReady || (bgColor === 'bg-yellow-100') || !enableProcess;
    }
}

/** Shows a step section by ID and hides others. */
function showStep(stepId) {
    for (let i = 1; i <= 8; i++) {
        const step = document.getElementById(`step-${i}`);
        if (step) {
            // Use block to ensure the card is still visible, but content is hidden/shown
            step.style.display = (step.id === stepId) ? 'block' : 'none';
        }
    }
}

/** Displays a message in a specified output div. */
function displayOutput(id, message, append = false) {
    const el = document.getElementById(id);
    if (el) {
        if (append) {
            el.innerHTML += message;
        } else {
            el.innerHTML = message;
        }
    }
}

/** Simple tokenizer: converts text to lowercase and splits by non-word characters. */
function simpleTokenizer(text) {
    if (!text || typeof text !== 'string') return [];
    return String(text).toLowerCase()
               .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,"")
               .split(/\s+/).filter(word => word.length > 0);
}

// --- STEP 1: DATA LOADING AND VALIDATION ---

/** Handles file input changes and validates file names. */
function handleFileChange(event) {
    const input = event.target;
    const file = input.files[0];

    const mapping = fileMappings.find(m => m.id === input.id);
    if (!file || !mapping) {
        // This case should be rare if user just selected a file and then changed their mind
        return; 
    } 

    if (file.name === mapping.expectedName) {
        updateFileStatus(mapping.statusId, `✅ Loaded: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`, 'success');
        loadedFiles[mapping.id.replace('File', '')] = file;
    } else {
        updateFileStatus(
            mapping.statusId,
            `❌ Name mismatch. Uploaded: "${file.name}". Expected: "${mapping.expectedName}".`,
            'error'
        );
        input.value = ''; 
        loadedFiles[mapping.id.replace('File', '')] = null;
    }

    const allFilesReady = Object.values(loadedFiles).every(f => f !== null);
    updateGeneralStatus(
        allFilesReady ? "All files successfully loaded. Ready to parse." : "Please ensure all three files are selected and named correctly.",
        allFilesReady ? 'bg-green-100' : 'bg-gray-100',
        allFilesReady ? 'text-green-800' : 'text-gray-600',
        allFilesReady // Enable button if files are ready
    );
}

/** Loads the content of the validated CSV files using PapaParse. */
function loadData() {
    updateGeneralStatus("Parsing CSV files...", 'bg-yellow-100', 'text-yellow-800', false);

    const keys = Object.keys(loadedFiles);
    let filesParsed = 0;

    keys.forEach(key => {
        const file = loadedFiles[key];

        Papa.parse(file, {
            header: true,
            dynamicTyping: true, // Let PapaParse try to infer types
            skipEmptyLines: true,
            complete: function(results) {
                // *** FIX APPLIED HERE: We are storing raw data without validation to allow 
                // the user to inspect in Step 2, even if it has errors. ***
                rawParsedData[key] = results.data;
                filesParsed++;
                
                if (filesParsed === keys.length) {
                    const emptyKeys = keys.filter(key => rawParsedData[key].length === 0);

                    if (emptyKeys.length > 0) {
                         // Halt and provide actionable error if files were truly empty/malformed
                        updateGeneralStatus(
                            `❌ Critical File Read Error. The following dataset(s) still resulted in 0 rows after parsing: **${emptyKeys.join(', ')}**. Please confirm the CSV files are valid and contain data.`, 
                            'bg-red-100', 
                            'text-red-800', 
                            true // Re-enable button to allow retrying
                        );
                        showStep('step-1'); 
                    } else {
                        // SUCCESS PATH
                        updateGeneralStatus(`✅ All data successfully parsed. Training: ${rawParsedData.training.length} rows. Ready for inspection.`, 'bg-green-100', 'text-green-800', true);
                        document.getElementById('inspectBtn').disabled = false;
                        showStep('step-2');
                    }
                }
            },
            error: function(error) {
                updateGeneralStatus(`❌ Error parsing ${file.name}: ${error.message}`, 'bg-red-100', 'text-red-800', true);
            }
        });
    });
}


// --- STEP 2: DATA INSPECTION ---

/** Inspects the loaded data, normalizes columns, and validates labels. */
function inspectData() {
    // Start by assuming training data is the one to inspect visually
    const dataKey = 'training';
    const rawData = rawParsedData[dataKey];

    if (!rawData || rawData.length === 0) {
        displayOutput('inspectionMessage', 'Error: Training data is empty or invalid. Please check the columns in your CSV files.', 'error');
        displayOutput('inspectionOutput', 'No data to show.', false);
        return;
    }

    // 1. Infer and normalize headers
    const firstRow = rawData[0];
    const rowKeys = Object.keys(firstRow);
    
    // Find text and label keys (case-insensitive and robust to extra spaces/quotes)
    let textKey = rowKeys.find(k => String(k).toLowerCase().trim().replace(/['"]/g, '') === 'text');
    let labelKey = rowKeys.find(k => String(k).toLowerCase().trim().replace(/['"]/g, '') === 'label');
    
    if (!textKey || !labelKey) {
        // If not found, fall back to the first two keys found
        const keys = Object.keys(firstRow).filter(k => String(k).trim().length > 0);
        textKey = textKey || keys[0];
        labelKey = labelKey || keys[1];
        
        displayOutput('inspectionMessage', `
            ⚠️ **WARNING**: Could not find standard 'text' and 'label' columns. 
            Inferring columns as **${textKey}** (input) and **${labelKey}** (output). 
            Please ensure these are correct!
            <br>
            Showing first 5 rows of the Training Set (${rawData.length} total rows). 
        `);
    } else {
        displayOutput('inspectionMessage', `Showing first 5 rows of the Training Set (${rawData.length} total rows). Columns: **${textKey}** (input) and **${labelKey}** (0=AI, 1=Human).`);
    }

    // 2. Normalize and Validate ALL Data
    const keysToNormalize = Object.keys(rawParsedData);
    let totalInvalidRows = 0;

    keysToNormalize.forEach(key => {
        const dataToNormalize = rawParsedData[key];
        let invalidRows = 0;
        
        normalizedData[key] = dataToNormalize.map(row => {
            const text = row[textKey];
            const rawLabel = row[labelKey];
            let label = null;

            // Robust Label Conversion: Handles numeric 0/1, string '0'/'1', and converts to actual number (0 or 1)
            // Fix: Ensure we can handle all common representations of 0 and 1
            if (rawLabel === 0 || rawLabel === 1) {
                label = rawLabel;
            } else if (typeof rawLabel === 'string') {
                const trimmed = rawLabel.trim();
                if (trimmed === '0') label = 0;
                else if (trimmed === '1') label = 1;
            } else if (typeof rawLabel === 'number' && (rawLabel === 0 || rawLabel === 1)) {
                label = rawLabel;
            }
            
            // Validate: check if text is present and label is a valid 0 or 1
            if (typeof text !== 'string' || String(text).trim().length === 0 || label === null) {
                invalidRows++;
                return null; // Exclude invalid rows
            }

            return { text: String(text).trim(), label: label };
        }).filter(row => row !== null);
        
        totalInvalidRows += invalidRows;
    });

    if (totalInvalidRows > 0) {
        // Alert user about rows being excluded
        updateGeneralStatus(
            `⚠️ Data Cleaning Complete. A total of **${totalInvalidRows}** rows were excluded across all datasets because they were missing text or had an invalid label (not 0 or 1). Proceeding with ${normalizedData.training.length} training rows.`,
            'bg-yellow-100',
            'text-yellow-800',
            false // Proceed to next step
        );
    } else {
        updateGeneralStatus(
            `✅ All data successfully inspected and cleaned. Ready for preprocessing.`,
            'bg-green-100',
            'text-green-800',
            false // Proceed to next step
        );
    }
    
    // 3. Display data table using the keys found/inferred
    let tableHtml = `<div class="overflow-x-auto"><table class="min-w-full divide-y divide-gray-200"><thead><tr>`;
    const headers = [textKey, labelKey]; 
    headers.forEach(h => tableHtml += `<th class="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase">${h}</th>`);
    tableHtml += `</tr></thead><tbody class="divide-y divide-gray-200">`;
    
    // Use the normalized data (with 'text' and 'label' keys) for displaying the content
    const dataToShow = normalizedData[dataKey].length > 0 ? normalizedData[dataKey] : rawData;
    dataToShow.slice(0, 5).forEach(row => {
        tableHtml += `<tr>`;
        // Check if the data is normalized (uses .text) or raw (uses [textKey])
        const textValue = row.text ? row.text : row[textKey] || 'N/A';
        const labelValue = row.label !== undefined && row.label !== null ? row.label : row[labelKey] || 'N/A';

        tableHtml += `<td class="px-3 py-3 text-sm text-gray-900 w-3/4 max-w-xs overflow-hidden text-ellipsis whitespace-nowrap">${textValue}</td>`;
        tableHtml += `<td class="px-3 py-3 whitespace-nowrap text-sm font-bold text-gray-900">${labelValue}</td>`;
        tableHtml += `</tr>`;
    });
    tableHtml += `</tbody></table></div>`;
    displayOutput('inspectionOutput', tableHtml);

    // Enable next step
    document.getElementById('preprocessBtn').disabled = false;
    showStep('step-3');
}


// --- STEP 3: PREPROCESSING (Tokenization & Vocabulary) ---

/** Tokenizes text and builds the global word-to-index map. */
function preprocessData() {
    document.getElementById('preprocessBtn').disabled = true;
    displayOutput('preprocessOutput', 'Building vocabulary from training data... <br>', false);
    
    // 1. Collect all unique words
    const uniqueWords = new Set();
    normalizedData.training.forEach(row => {
        const tokens = simpleTokenizer(row.text); 
        tokens.forEach(word => uniqueWords.add(word));
    });

    // 2. Create word-to-index map (index starts at 2, 0 and 1 are reserved)
    let index = 2; 
    wordIndex = { '<PAD>': 0, '<OOV>': 1 }; // OOV (Out-Of-Vocabulary) placeholder at index 1
    uniqueWords.forEach(word => {
        if (!wordIndex[word]) {
            wordIndex[word] = index++;
        }
    });

    VOCAB_SIZE = index;

    displayOutput('preprocessOutput', '✅ Vocabulary built: <br>' +
        `Total Unique Words Found: **${uniqueWords.size}** <br>` +
        `Vocabulary Size (including PAD/OOV): **${VOCAB_SIZE}** <br>` +
        `Max Sequence Length for Padding: **${MAX_SEQUENCE_LENGTH}**`, true);
    
    document.getElementById('maxSeqLenDisplay').textContent = MAX_SEQUENCE_LENGTH;
    
    // Enable next step
    document.getElementById('embeddingBtn').disabled = false;
    showStep('step-4');
}


// --- STEP 4: TEXT EMBEDDING (Sequencing & Padding) ---

/** Converts raw text data into padded numerical sequences (Tensors). */
function createEmbeddings() {
    document.getElementById('embeddingBtn').disabled = true;
    displayOutput('embeddingOutput', 'Converting text to padded sequences and Tensors... <br>', false);
    
    // Global variable to hold the final processed Tensors for training
    const processedTensors = {};

    const processTextToSequence = (text, wordIndexMap, maxLength) => {
        const tokens = simpleTokenizer(text);
        // Map tokens to indices, using 1 ('<OOV>') for unknown words
        let sequence = tokens.map(word => wordIndexMap[word] || 1); 

        // Apply truncation (if sequence is longer than max length)
        if (sequence.length > maxLength) {
            sequence = sequence.slice(0, maxLength);
        }
        // Apply padding (if sequence is shorter than max length)
        while (sequence.length < maxLength) {
            sequence.push(0); // 0 is '<PAD>'
        }
        return sequence;
    };
    
    try {
        ['training', 'testing', 'validation'].forEach(key => {
            const data = normalizedData[key];

            const sequences = data.map(row => processTextToSequence(row.text, wordIndex, MAX_SEQUENCE_LENGTH));
            const labels = data.map(row => row.label);

            const featureTensor = tf.tensor2d(sequences, [data.length, MAX_SEQUENCE_LENGTH], 'int32');
            const labelTensor = tf.tensor2d(labels, [data.length, 1], 'int32');

            processedTensors[key] = { features: featureTensor, labels: labelTensor };

            displayOutput('embeddingOutput', 
                `**${key.toUpperCase()}** - Samples: ${data.length}, Sequences Shape: ${featureTensor.shape} <br> `, true);

        });
            
        // Overwrite the global normalizedData with the tensors
        Object.assign(normalizedData, processedTensors);

        displayOutput('embeddingOutput', '✅ All datasets successfully converted to numerical sequences.', true);

        // Enable next step
        document.getElementById('createModelBtn').disabled = false;
        showStep('step-5');

    } catch (error) {
        displayOutput('embeddingOutput', `❌ Embedding failed: ${error.message}`, true);
        document.getElementById('embeddingBtn').disabled = false;
    }
}


// --- STEP 5: MODEL SETUP ---

/** Defines and compiles the text classification neural network model. */
function createModel() {
    document.getElementById('createModelBtn').disabled = true;
    
    // Input shape is (MAX_SEQUENCE_LENGTH)
    model = tf.sequential();
    
    // 1. Embedding Layer: Turns word indices into dense vectors
    model.add(tf.layers.embedding({
        inputDim: VOCAB_SIZE,        // Vocabulary size
        outputDim: EMBEDDING_DIM,    // Embedding vector size (e.g., 16)
        inputLength: MAX_SEQUENCE_LENGTH // Padded sequence length
    }));
    
    // 2. Flatten the embedded sequences (from [50, 16] to [800])
    model.add(tf.layers.flatten());

    // 3. Dense layers for classification
    model.add(tf.layers.dense({ units: 10, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 10, activation: 'relu' }));
    
    // 4. Output layer (Binary Classification: 1 or 0)
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' })); 

    // Compile the model
    model.compile({
        optimizer: tf.train.adam(0.005),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    // Display model summary
    let summaryText = 'Model Architecture: <br>';
    const originalLog = console.log;
    // Intercept console.log to capture model summary output
    console.log = (message) => { summaryText += message.replace(/\n/g, '<br>') + '<br>'; };
    model.summary();
    console.log = originalLog; // Restore original console.log
    displayOutput('modelSummary', summaryText);

    // Enable next step
    document.getElementById('trainModelBtn').disabled = false;
    showStep('step-6');
}


// --- STEP 6: MODEL TRAINING ---

/** Trains the defined model using the training data. */
async function trainModel() {
    if (!model || !normalizedData.training || !normalizedData.training.features) {
        displayOutput('trainingOutput', 'Model or data not ready. Please complete previous steps.', false);
        return;
    }

    const epochs = parseInt(document.getElementById('epochsInput').value, 10);
    const trainFeatures = normalizedData.training.features;
    const trainLabels = normalizedData.training.labels;

    document.getElementById('trainModelBtn').disabled = true;
    displayOutput('trainingOutput', 'Training started... See visualization below.');

    // Prepare container for tfjs-vis
    const historyContainer = document.getElementById('trainingVisContainer');
    historyContainer.innerHTML = '';
    const container = { name: 'Training Metrics', tab: 'Training' };
    const metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy'];
    const callbacks = tfvis.show.fitCallbacks(container, metrics);

    try {
        const history = await model.fit(trainFeatures, trainLabels, {
            batchSize: 32,
            epochs: epochs,
            validationData: [normalizedData.validation.features, normalizedData.validation.labels],
            callbacks: callbacks
        });

        const finalLoss = history.history.loss.slice(-1)[0].toFixed(4);
        const finalValAcc = history.history.val_accuracy.slice(-1)[0].toFixed(4);
        displayOutput('trainingOutput', `✅ Training finished after ${history.params.epochs} epochs. Final Training Loss: ${finalLoss}, Final Validation Accuracy: ${finalValAcc}.`);
        
        // Enable next step
        document.getElementById('evaluateBtn').disabled = false;
        showStep('step-7');

    } catch (error) {
        displayOutput('trainingOutput', `❌ Training failed: ${error.message}`, false);
        document.getElementById('trainModelBtn').disabled = false;
    }
}


// --- STEP 7: MODEL EVALUATION ---

/** Evaluates the model on the validation dataset. */
async function evaluateModel() {
    if (!model || !normalizedData.validation) {
        displayOutput('evaluationOutput', 'Model or validation data not ready.', false);
        return;
    }
    document.getElementById('evaluateBtn').disabled = true;

    displayOutput('evaluationOutput', 'Evaluating model on validation data...');

    const evalResult = model.evaluate(normalizedData.validation.features, normalizedData.validation.labels);
    // evalResult is an array of Tensors (loss, accuracy). We need to pull the value out.
    const [loss, accuracy] = await Promise.all(evalResult.map(t => t.data()));

    displayOutput('evaluationOutput', `
        ✅ Evaluation Complete. <br>
        <strong>Validation Loss:</strong> ${loss.toFixed(4)} <br>
        <strong>Validation Accuracy:</strong> ${accuracy.toFixed(4)}
    `);
    
    // Enable next step
    document.getElementById('predictBtn').disabled = false;
    showStep('step-8');
}


// --- STEP 8: PREDICTION VALUE ---

/** Generates a prediction for user-supplied text. */
async function makePrediction() {
    if (!model) {
        displayOutput('predictionOutput', 'Model is not trained. Please complete all previous steps.', false);
        return;
    }

    const inputText = document.getElementById('predictionText').value.trim();
    if (!inputText) {
        displayOutput('predictionOutput', 'Please enter text to analyze.', false);
        return;
    }

    document.getElementById('predictBtn').disabled = true;
    displayOutput('predictionOutput', 'Analyzing text...');

    // 1. Preprocess the user's text (Tokenize, map to index, pad)
    const tokens = simpleTokenizer(inputText);
    let sequence = tokens.map(word => wordIndex[word] || 1); // Use 1 for OOV

    if (sequence.length > MAX_SEQUENCE_LENGTH) {
        sequence = sequence.slice(0, MAX_SEQUENCE_LENGTH);
    }
    while (sequence.length < MAX_SEQUENCE_LENGTH) {
        sequence.push(0); 
    }

    // 2. Convert to Tensor (shape [1, MAX_SEQUENCE_LENGTH])
    const inputTensor = tf.tensor2d([sequence], [1, MAX_SEQUENCE_LENGTH], 'int32');

    // 3. Generate prediction
    const predictionTensor = model.predict(inputTensor);
    // Await the data() call to get the probability value
    const probabilityArray = await predictionTensor.data();
    const probability = probabilityArray[0]; // Probability of class 1 (Human)

    // 4. Format and display results
    const humanProbability = (probability * 100).toFixed(2);
    const aiProbability = ((1 - probability) * 100).toFixed(2);
    
    let resultMessage = `Prediction Complete: <br>`;
    
    if (probability > 0.5) {
        resultMessage += `<strong class="text-green-800">The model predicts this is LIKELY HUMAN-WRITTEN.</strong><br>`;
    } else {
        resultMessage += `<strong class="text-red-800">The model predicts this is LIKELY AI-GENERATED.</strong><br>`;
    }
    
    resultMessage += `<br>`;
    resultMessage += `Human-Written Probability: **${humanProbability}%** <br>`;
    resultMessage += `AI-Generated Probability: **${aiProbability}%**`;

    displayOutput('predictionOutput', resultMessage);

    // Clean up
    inputTensor.dispose();
    predictionTensor.dispose();
    document.getElementById('predictBtn').disabled = false;
}


// --- INITIALIZATION ---

document.addEventListener('DOMContentLoaded', () => {
    // 1. Attach the change handler to all file inputs
    fileMappings.forEach(mapping => {
        const inputElement = document.getElementById(mapping.id);
        if (inputElement) {
            inputElement.addEventListener('change', handleFileChange);
            updateFileStatus(mapping.statusId, `Awaiting file "${mapping.expectedName}"...`, 'info');
        }
    });

    // 2. Attach click handlers
    document.getElementById('processBtn').addEventListener('click', loadData);
    document.getElementById('inspectBtn').addEventListener('click', inspectData);
    document.getElementById('preprocessBtn').addEventListener('click', preprocessData);
    document.getElementById('embeddingBtn').addEventListener('click', createEmbeddings);
    document.getElementById('createModelBtn').addEventListener('click', createModel);
    document.getElementById('trainModelBtn').addEventListener('click', trainModel);
    document.getElementById('evaluateBtn').addEventListener('click', evaluateModel);
    document.getElementById('predictBtn').addEventListener('click', makePrediction);
    
    // 3. Set initial state for all subsequent steps (2 through 8)
    // The HTML is structured to hide these by default, but this ensures JS state consistency
    for (let i = 2; i <= 8; i++) {
        const step = document.getElementById(`step-${i}`);
        if (step) step.style.display = 'none';
    }
    showStep('step-1');
});
