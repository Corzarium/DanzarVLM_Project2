document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const commentaryToggle = document.getElementById('commentaryToggle');
    const ocrToggle = document.getElementById('ocrToggle');
    const imageFormat = document.getElementById('imageFormat');
    const imageSize = document.getElementById('imageSize');
    const captureScreenshot = document.getElementById('captureScreenshot');
    const minInterval = document.getElementById('minInterval');
    const maxInterval = document.getElementById('maxInterval');
    const updateIntervalsBtn = document.getElementById('updateIntervals');
    const messageInput = document.getElementById('messageInput');
    const sendMessageBtn = document.getElementById('sendMessage');
    const chatMessages = document.getElementById('chatMessages');
    const statusMessage = document.getElementById('statusMessage');
    const refreshPrompts = document.getElementById('refreshPrompts');
    // New elements for image upload and local path
    const imageUpload = document.getElementById('imageUpload');
    const analyzeImageFileBtn = document.getElementById('analyzeImageFile');
    const localImagePath = document.getElementById('localImagePath');
    const analyzeLocalPathBtn = document.getElementById('analyzeLocalPath');
    // New diagnostic button
    const runDiagnosticBtn = document.getElementById('runDiagnosticBtn');
    
    // Prompt display elements
    const systemPromptDisplay = document.getElementById('systemPromptDisplay');
    const ocrEnabledPromptDisplay = document.getElementById('ocrEnabledPromptDisplay');
    const ocrDisabledPromptDisplay = document.getElementById('ocrDisabledPromptDisplay');
    const chatPromptDisplay = document.getElementById('chatPromptDisplay');

    // Initialize state
    fetchCurrentState();
    
    // Fetch prompts initially
    fetchPrompts();

    // Event Listeners
    commentaryToggle.addEventListener('change', function() {
        toggleCommentary(this.checked);
    });

    ocrToggle.addEventListener('change', function() {
        toggleOCR(this.checked);
    });
    
    imageFormat.addEventListener('change', function() {
        setImageFormat(this.value);
    });
    
    imageSize.addEventListener('change', function() {
        setImageSize(this.value);
    });
    
    captureScreenshot.addEventListener('click', function() {
        this.disabled = true;
        this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Capturing...';
        captureAndAnalyzeScreenshot();
    });
    
    // Add event listener for image upload button
    if (analyzeImageFileBtn) {
        analyzeImageFileBtn.addEventListener('click', function() {
            uploadAndAnalyzeImage();
        });
    }
    
    // Add event listener for local path analysis button
    if (analyzeLocalPathBtn) {
        analyzeLocalPathBtn.addEventListener('click', function() {
            analyzeLocalImagePath();
        });
    }
    
    // Add event listener for diagnostic button
    if (runDiagnosticBtn) {
        runDiagnosticBtn.addEventListener('click', function() {
            runLLMVisionDiagnostic();
        });
    }
    
    refreshPrompts.addEventListener('click', function() {
        this.disabled = true;
        this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Refreshing...';
        fetchPrompts()
            .then(() => {
                this.disabled = false;
                this.innerHTML = 'Refresh Prompts';
                updateStatus('Prompts refreshed successfully', 'success');
            })
            .catch(error => {
                this.disabled = false;
                this.innerHTML = 'Refresh Prompts';
                updateStatus('Error refreshing prompts', 'danger');
            });
    });

    updateIntervalsBtn.addEventListener('click', function() {
        updateIntervals();
    });

    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    sendMessageBtn.addEventListener('click', sendMessage);

    // Functions
    function fetchCurrentState() {
        fetch('/api/state')
            .then(response => response.json())
            .then(data => {
                commentaryToggle.checked = data.commentary_enabled;
                ocrToggle.checked = data.ocr_enabled;
                minInterval.value = data.min_interval;
                maxInterval.value = data.max_interval;
                
                // Set the image format dropdown
                if (data.image_format && imageFormat) {
                    imageFormat.value = data.image_format;
                }
                
                // Set image size if it's available
                if (data.max_image_size && imageSize) {
                    imageSize.value = data.max_image_size;
                }
                
                updateStatus('System connected and ready');
            })
            .catch(error => {
                console.error('Error fetching state:', error);
                updateStatus('Error connecting to server', 'danger');
            });
    }

    function toggleCommentary(enabled) {
        fetch('/api/commentary', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ enabled: enabled })
        })
        .then(response => response.json())
        .then(data => {
            updateStatus(`Commentary ${enabled ? 'enabled' : 'disabled'}`, 'success');
        })
        .catch(error => {
            console.error('Error toggling commentary:', error);
            updateStatus('Error toggling commentary', 'danger');
            commentaryToggle.checked = !enabled; // Revert toggle
        });
    }

    function toggleOCR(enabled) {
        fetch('/api/ocr', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ enabled: enabled })
        })
        .then(response => response.json())
        .then(data => {
            if (enabled) {
                updateStatus('OCR Processing enabled', 'success');
            } else {
                const imageFormat = data.image_format || 'markdown';
                updateStatus(`OCR disabled. Using image format: ${imageFormat}`, 'info');
            }
        })
        .catch(error => {
            console.error('Error toggling OCR:', error);
            updateStatus('Error toggling OCR', 'danger');
            ocrToggle.checked = !enabled; // Revert toggle
        });
    }
    
    function setImageFormat(format) {
        fetch('/api/image-format', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ format: format })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateStatus(`Image format set to: ${format}`, 'success');
            } else {
                updateStatus(data.error || 'Error setting image format', 'warning');
            }
        })
        .catch(error => {
            console.error('Error setting image format:', error);
            updateStatus('Error updating image format', 'danger');
        });
    }

    function setImageSize(size) {
        fetch('/api/image-size', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ max_size: size })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateStatus(`Image size set to: ${size}px`, 'success');
            } else {
                updateStatus(data.error || 'Error setting image size', 'warning');
            }
        })
        .catch(error => {
            console.error('Error setting image size:', error);
            updateStatus('Error updating image size', 'danger');
        });
    }
    
    function captureAndAnalyzeScreenshot() {
        // Get current image size
        const size = imageSize.value;
        
        fetch('/api/capture-screenshot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ max_size: size })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateStatus(`Screenshot captured and sent for analysis (${data.image_size}px, ${data.image_format} format)`, 'success');
            } else {
                updateStatus(data.error || 'Error capturing screenshot', 'warning');
            }
            
            // Re-enable the button
            captureScreenshot.disabled = false;
            captureScreenshot.innerHTML = 'Capture Screenshot & Analyze';
        })
        .catch(error => {
            console.error('Error capturing screenshot:', error);
            updateStatus('Error processing screenshot', 'danger');
            
            // Re-enable the button
            captureScreenshot.disabled = false;
            captureScreenshot.innerHTML = 'Capture Screenshot & Analyze';
        });
    }

    function updateIntervals() {
        const minVal = parseInt(minInterval.value);
        const maxVal = parseInt(maxInterval.value);

        if (minVal >= maxVal) {
            updateStatus('Minimum interval must be less than maximum interval', 'warning');
            return;
        }

        fetch('/api/intervals', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                min_interval: minVal,
                max_interval: maxVal
            })
        })
        .then(response => response.json())
        .then(data => {
            updateStatus('Intervals updated successfully', 'success');
        })
        .catch(error => {
            console.error('Error updating intervals:', error);
            updateStatus('Error updating intervals', 'danger');
        });
    }

    function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;

        // Add user message to chat
        addMessageToChat('user', message);
        messageInput.value = '';

        // Send to server
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                addMessageToChat('bot', data.response);
            } else {
                updateStatus(data.error || 'Error receiving response', 'danger');
            }
        })
        .catch(error => {
            console.error('Error sending message:', error);
            updateStatus('Error sending message', 'danger');
        });
    }

    function addMessageToChat(type, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        // Format the content for better readability
        content = content.replace(/\*\*/g, '') // Remove Discord-style bold markers
                        .replace(/[*#]/g, '')   // Remove asterisks and hashes
                        .trim();
        
        messageDiv.textContent = content;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function updateStatus(message, type = 'info') {
        statusMessage.className = `alert alert-${type}`;
        statusMessage.textContent = message;
    }

    // Fetch and display prompts
    function fetchPrompts() {
        return fetch('/api/prompts')
            .then(response => response.json())
            .then(data => {
                // Format and display the prompts
                if (systemPromptDisplay) {
                    systemPromptDisplay.textContent = data.system_prompt_commentary || 'System prompt not defined';
                }
                
                if (ocrEnabledPromptDisplay) {
                    ocrEnabledPromptDisplay.textContent = data.user_prompt_template_commentary_ocr || 'OCR enabled prompt not defined';
                }
                
                if (ocrDisabledPromptDisplay) {
                    ocrDisabledPromptDisplay.textContent = data.ocr_disabled_template || 'OCR disabled prompt not defined';
                }
                
                if (chatPromptDisplay) {
                    chatPromptDisplay.textContent = data.system_prompt_chat || 'Chat system prompt not defined';
                }
            })
            .catch(error => {
                console.error('Error fetching prompts:', error);
                updateStatus('Error fetching prompts', 'danger');
            });
    }

    // New function for uploading and analyzing an image file
    function uploadAndAnalyzeImage() {
        // Check if a file is selected
        if (!imageUpload || !imageUpload.files || imageUpload.files.length === 0) {
            updateStatus('Please select an image file first', 'warning');
            return;
        }
        
        // Get the selected file
        const file = imageUpload.files[0];
        
        // Get current image size setting
        const size = imageSize.value;
        
        // Create a FormData object to send the file
        const formData = new FormData();
        formData.append('file', file);
        formData.append('max_size', size);
        
        // Show loading state
        analyzeImageFileBtn.disabled = true;
        analyzeImageFileBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Uploading...';
        
        // Send the file to the server
        fetch('/api/upload-image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateStatus(`Image uploaded and analyzed (${data.image_size}px, ${data.image_format} format, ${data.shape})`, 'success');
            } else {
                updateStatus(data.error || 'Error analyzing image', 'warning');
            }
            
            // Re-enable the button
            analyzeImageFileBtn.disabled = false;
            analyzeImageFileBtn.innerHTML = 'Upload & Analyze Image';
        })
        .catch(error => {
            console.error('Error uploading image:', error);
            updateStatus('Error processing image upload', 'danger');
            
            // Re-enable the button
            analyzeImageFileBtn.disabled = false;
            analyzeImageFileBtn.innerHTML = 'Upload & Analyze Image';
        });
    }
    
    // New function for analyzing an image from a local path
    function analyzeLocalImagePath() {
        // Check if a path is entered
        const path = localImagePath.value.trim();
        if (!path) {
            updateStatus('Please enter a valid image path', 'warning');
            return;
        }
        
        // Get current image size setting
        const size = imageSize.value;
        
        // Show loading state
        analyzeLocalPathBtn.disabled = true;
        analyzeLocalPathBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
        
        // Send the path to the server
        fetch('/api/local-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                path: path,
                max_size: size
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateStatus(`Local image analyzed (${data.image_size}px, ${data.image_format} format, ${data.shape})`, 'success');
            } else {
                updateStatus(data.error || 'Error analyzing local image', 'warning');
            }
            
            // Re-enable the button
            analyzeLocalPathBtn.disabled = false;
            analyzeLocalPathBtn.innerHTML = 'Analyze';
        })
        .catch(error => {
            console.error('Error analyzing local image:', error);
            updateStatus('Error processing local image', 'danger');
            
            // Re-enable the button
            analyzeLocalPathBtn.disabled = false;
            analyzeLocalPathBtn.innerHTML = 'Analyze';
        });
    }

    // New function for running the LLM vision diagnostic test
    function runLLMVisionDiagnostic() {
        // Get current image size setting
        const size = imageSize.value;
        
        // Show loading state
        runDiagnosticBtn.disabled = true;
        runDiagnosticBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Running Tests...';
        
        // Send request to run diagnostics
        fetch('/api/run-diagnostic', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                source: 'latest',
                max_size: size
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateStatus(`Diagnostic test started - testing all image formats (${data.image_size}px). Check chat for results.`, 'info');
                
                // Add a message in the chat area to explain
                addMessageToChat('system', 'Running vision diagnostics. Testing all image formats to find one that works with your model. Results will be shown in this chat and saved to debug_vlm_frames directory.');
            } else {
                updateStatus(data.error || 'Error running diagnostic', 'warning');
            }
            
            // Re-enable the button after a short delay
            setTimeout(() => {
                runDiagnosticBtn.disabled = false;
                runDiagnosticBtn.innerHTML = 'Run Format Diagnostic';
            }, 3000);
        })
        .catch(error => {
            console.error('Error running diagnostic:', error);
            updateStatus('Error running vision diagnostic', 'danger');
            
            // Re-enable the button
            runDiagnosticBtn.disabled = false;
            runDiagnosticBtn.innerHTML = 'Run Format Diagnostic';
        });
    }

    // Optional: Setup WebSocket for real-time updates
    // This can be implemented later for real-time status updates
}); 