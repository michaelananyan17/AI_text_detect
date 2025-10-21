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

const MAX_SEQUENCE_LENGTH = 50; // Fixed sequence length for padding
const EMBEDDING_DIM = 16;       // Fixed size for the embedding vector

const loadedFiles = { training: null, testing: null, validation: null };
const rawData = { training: null, testing: null, validation: null };
const processedData = {}; 
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
        generalStatus.className = `mt-4 p-3 rounded-lg text-center font-medium transition-all duration-300 ${bgColor} ${textColor}`;
        generalStatus.textContent = message;
    }
    
    if (processBtn) {
        // Only disable the button if not all files are loaded, unless a data parsing error occurred (bg-yellow-100 is for processing)
        const allFilesReady = Object.values(loadedFiles).every(f => f !== null);
        processBtn.disabled = !allFilesReady || (bgColor === 'bg-yellow-100') || !enableProcess;
    }
}

/** Shows a step section by ID and hides others. */
function showStep(stepId) {
    for (let i = 1; i <= 8; i++) {
        const step = document.getElementById(`step-${i}`);
        if (step) {
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
    return text.toLowerCase()
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
        updateFileStatus(mapping.statusId, 'File not selected.', 'error');
        loadedFiles[mapping.id.replace('File', '')] = null;
    } else if (file.name === mapping.expectedName) {
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
        allFilesReady ? "All files successfully loaded. Ready to process." : "Please ensure all three files are selected and named correctly.",
        allFilesReady ? 'bg-green-100' : 'bg-gray-100',
        allFilesReady ? 'text-green-800' : 'text-gray-600',
        allFilesReady
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
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: function(results) {
                // Ensure text and label columns exist and filter invalid data
                // The label is checked against '0' and '1' in case PapaParse reads it as a string
                rawData[key] = results.data.filter(d => 
                    d.text !== undefined && d.text !== null && 
                    d.label !== undefined && d.label !== null && 
                    (d.label === 0 || d.label === 1 || d.label === '0' || d.label === '1') // Enforce binary label and handle string/number
                );
                filesParsed++;
                
                if (filesParsed === keys.length) {
                    // Check if any dataset is empty after filtering
                    const emptyKeys = keys.filter(key => rawData[key].length === 0);

                    if (emptyKeys.length > 0) {
                         // Halt and provide actionable error
                        updateGeneralStatus(
                            `❌ Data Validation Error. The following dataset(s) contained 0 valid rows after parsing: ${emptyKeys.join(', ')}. Please ensure the CSV files have columns named 'text' and 'label', and that 'label' contains only 0 or 1.`, 
                            'bg-red-100', 
                            'text-red-800', 
                            true // Re-enable button to allow retrying with corrected data
                        );
                        // Reset visibility to Step 1
                        showStep('step-1'); 
                    } else {
                        // SUCCESS PATH
                        updateGeneralStatus(`✅ All data successfully parsed. Training: ${rawData.training.length} rows. Ready for inspection.`, 'bg-green-100', 'text-green-800', false);
                        document.getElementById('step-2').style.display = 'block';
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

/** Inspects the loaded data (using training set) and shows statistics. */
function inspectData() {
    const data = rawData.training;
    if (!data || data.length === 0) {
        displayOutput('inspectionMessage', 'Error: Training data is empty or invalid. Please check the `text` and `label` columns in your CSV files.', 'error');
        displayOutput('inspectionOutput', '', false);
        return;
    }

    displayOutput('inspectionMessage', `Showing first 5 rows of the Training Set (${data.length} total rows). Columns: **text** (input) and **label** (0=AI, 1=Human).`);
    
    // Display data table
    let tableHtml = `<div class="overflow-x-auto"><table class="min-w-full divide-y divide-gray-200"><thead><tr>`;
    // Only show text and label for inspection clarity
    const headers = ['text', 'label']; 
    headers.forEach(h => tableHtml += `<th class="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase">${h}</th>`);
    tableHtml += `</tr></thead><tbody class="divide-y divide-gray-200">`;
    
    data.slice(0, 5).forEach(row => {
        tableHtml += `<tr>`;
        tableHtml += `<td class="px-3 py-3 text-sm text-gray-900 w-3/4 max-w-xs overflow-hidden text-ellipsis whitespace-nowrap">${row.text}</td>`;
        tableHtml += `<td class="px-3 py-3 whitespace-nowrap text-sm font-bold text-gray-900">${row.label}</td>`;
        tableHtml += `</tr>`;
    });
    tableHtml += `</tbody></table></div>`;
    displayOutput('inspectionOutput', tableHtml);

    // Enable next step
    document.getElementById('step-3').style.display = 'block';
    document.getElementById('preprocessBtn').disabled = false;
    showStep('step-3');
}


// --- STEP 3: PREPROCESSING (Tokenization & Vocabulary) ---

/** Tokenizes text and builds the global word-to-index map. */
function preprocessData() {
    displayOutput('preprocessOutput', 'Building vocabulary from training data... <br>', false);
    
    // 1. Collect all unique words
    const uniqueWords = new Set();
    rawData.training.forEach(row => {
        const tokens = simpleTokenizer(row.text);
        tokens.forEach(word => uniqueWords.add(word));
    });

    // 2. Create word-to-index map (index starts at 1, 0 is reserved for padding)
    let index = 1; 
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
    document.getElementById('step-4').style.display = 'block';
    document.getElementById('embeddingBtn').disabled = false;
    showStep('step-4');
}


// --- STEP 4: TEXT EMBEDDING (Sequencing & Padding) ---

/** Converts raw text data into padded numerical sequences (Tensors). */
function createEmbeddings() {
    displayOutput('embeddingOutput', 'Converting text to padded sequences and Tensors... <br>', false);

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
            const data = rawData[key];

            const sequences = data.map(row => processTextToSequence(row.text, wordIndex, MAX_SEQUENCE_LENGTH));
            const labels = data.map(row => row.label);

            const featureTensor = tf.tensor2d(sequences, [data.length, MAX_SEQUENCE_LENGTH], 'int32');
            const labelTensor = tf.tensor2d(labels, [data.length, 1], 'int32');

            processedData[key] = { features: featureTensor, labels: labelTensor };

            displayOutput('embeddingOutput', 
                `**${key.toUpperCase()}** - Sequences Shape: ${featureTensor.shape} <br> `, true);

        });
            
        displayOutput('embeddingOutput', '✅ All datasets successfully converted to numerical sequences.', true);

        // Enable next step
        document.getElementById('step-5').style.display = 'block';
        document.getElementById('createModelBtn').disabled = false;
        showStep('step-5');

    } catch (error) {
        displayOutput('embeddingOutput', `❌ Embedding failed: ${error.message}`, true);
    }
}


// --- STEP 5: MODEL SETUP ---

/** Defines and compiles the text classification neural network model. */
function createModel() {
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
    console.log = (message) => { summaryText += message.replace(/\n/g, '<br>') + '<br>'; };
    model.summary();
    console.log = originalLog; // Restore original console.log
    displayOutput('modelSummary', summaryText);

    // Enable next step
    document.getElementById('step-6').style.display = 'block';
    document.getElementById('trainModelBtn').disabled = false;
    showStep('step-6');
}


// --- STEP 6: MODEL TRAINING ---

/** Trains the defined model using the training data. */
async function trainModel() {
    if (!model || !processedData.training) {
        displayOutput('trainingOutput', 'Model or data not ready. Please complete previous steps.', false);
        return;
    }

    const epochs = parseInt(document.getElementById('epochsInput').value, 10);
    const trainFeatures = processedData.training.features;
    const trainLabels = processedData.training.labels;

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
            validationData: [processedData.validation.features, processedData.validation.labels],
            callbacks: callbacks
        });

        const finalLoss = history.history.loss.slice(-1)[0].toFixed(4);
        const finalValAcc = history.history.val_accuracy.slice(-1)[0].toFixed(4);
        displayOutput('trainingOutput', `✅ Training finished after ${history.params.epochs} epochs. Final Training Loss: ${finalLoss}, Final Validation Accuracy: ${finalValAcc}.`);
        
        // Enable next step
        document.getElementById('step-7').style.display = 'block';
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
    if (!model || !processedData.validation) {
        displayOutput('evaluationOutput', 'Model or validation data not ready.', false);
        return;
    }
    document.getElementById('evaluateBtn').disabled = true;

    displayOutput('evaluationOutput', 'Evaluating model on validation data...');

    const evalResult = model.evaluate(processedData.validation.features, processedData.validation.labels);
    const [loss, accuracy] = evalResult.map(t => t.dataSync()[0]);

    displayOutput('evaluationOutput', `
        ✅ Evaluation Complete. <br>
        <strong>Validation Loss:</strong> ${loss.toFixed(4)} <br>
        <strong>Validation Accuracy:</strong> ${accuracy.toFixed(4)}
    `);
    
    // Enable next step
    document.getElementById('step-8').style.display = 'block';
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
    const probability = predictionTensor.dataSync()[0]; // Probability of class 1 (Human)

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

    // 2. Attach the loadData function to the button click event
    const processBtn = document.getElementById('processBtn');
    if (processBtn) {
        processBtn.addEventListener('click', loadData);
    }
    
    // 3. Set initial state for all subsequent steps
    for (let i = 2; i <= 8; i++) {
        const step = document.getElementById(`step-${i}`);
        if (step) step.style.display = 'none';
    }
    showStep('step-1');
});
