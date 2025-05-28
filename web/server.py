from flask import Flask, render_template, jsonify, request
import threading
import os
import sys
import re
import numpy as np
import cv2

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

app = Flask(__name__)
app_context = None  # Will be set when server is started

def clean_text_for_tts(text):
    """Clean text for TTS by removing markdown and special characters."""
    # Remove Discord formatting
    text = text.replace('💬 **WebUI:** ', '').replace('🤖 **Reply:** ', '')
    # Remove markdown
    text = re.sub(r'\*\*|__|\*|#', '', text)
    # Remove excessive whitespace
    text = ' '.join(text.split())
    return text.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    if not app_context:
        return jsonify({'error': 'Server not initialized'}), 500
    
    # Check if we're using a Qwen model and set appropriate default
    model_name = getattr(app_context.active_profile, 'vlm_model', '').lower()
    if 'qwen' in model_name:
        # Set the preferred image format to qwen for Qwen models
        app_context.global_settings["VLM_IMAGE_FORMAT"] = "qwen"
        app_context.logger.info(f"Detected Qwen model: {model_name}. Setting image format to 'qwen'")
    elif app_context.global_settings.get("VLM_IMAGE_FORMAT") != "llava" and 'qwen' not in model_name:
        # For other models, default to llava format if not explicitly set
        app_context.global_settings["VLM_IMAGE_FORMAT"] = "llava"
        app_context.logger.info("Setting default image format to 'llava' for non-Qwen models")
    
    return jsonify({
        'commentary_enabled': app_context.ndi_commentary_enabled.is_set(),
        'ocr_enabled': getattr(app_context.active_profile, 'ocr_enabled', True),
        'min_interval': float(app_context.global_settings.get("NDI_MIN_RANDOM_COMMENTARY_INTERVAL_S", 30.0)),
        'max_interval': float(app_context.global_settings.get("NDI_MAX_RANDOM_COMMENTARY_INTERVAL_S", 120.0)),
        'image_format': app_context.global_settings.get("VLM_IMAGE_FORMAT", "llava"),
        'is_qwen': 'qwen' in model_name
    })

@app.route('/api/commentary', methods=['POST'])
def toggle_commentary():
    if not app_context:
        return jsonify({'error': 'Server not initialized'}), 500
    
    data = request.get_json()
    enabled = data.get('enabled', False)
    
    if enabled:
        app_context.ndi_commentary_enabled.set()
    else:
        app_context.ndi_commentary_enabled.clear()
    
    return jsonify({'success': True, 'enabled': enabled})

@app.route('/api/ocr', methods=['POST'])
def toggle_ocr():
    if not app_context:
        return jsonify({'error': 'Server not initialized'}), 500
    
    data = request.get_json()
    enabled = data.get('enabled', False)
    
    # Update the profile's OCR setting
    app_context.active_profile.ocr_enabled = enabled
    
    # If OCR is disabled, we should modify the LLM prompt to focus more on the visual content
    if not enabled and app_context.llm_service_instance:
        # Update the user prompt template to focus on visual analysis
        app_context.active_profile.user_prompt_template_commentary = (
            "Analyze the visual content of the game using the full screenshot provided. "
            "Focus on overall scene composition, character actions, environment changes, and notable events. "
            "Game: {game_name}. What interesting observations can you make about what you can see in the image?"
        )
        
        # Set debug mode to troubleshoot image visibility
        app_context.global_settings["VLM_DEBUG_MODE"] = True
        
        # Try different image formats to see which one works
        # Options: "markdown", "html", "llava", "cogvlm"
        current_format = app_context.global_settings.get("VLM_IMAGE_FORMAT", "markdown")
        formats = ["markdown", "llava", "html", "cogvlm"]
        
        # Cycle through formats
        if current_format in formats:
            next_index = (formats.index(current_format) + 1) % len(formats)
            next_format = formats[next_index]
        else:
            next_format = "llava"  # Try llava format next
            
        app_context.global_settings["VLM_IMAGE_FORMAT"] = next_format
        
        app_context.logger.warning(f"OCR disabled. TRYING NEW IMAGE FORMAT: {next_format}")
        app_context.logger.warning("FULL SCREENSHOTS WILL STILL BE SENT TO LLM for visual analysis.")
        
        # Force a log message in the LLM service as well
        if hasattr(app_context.llm_service_instance, 'logger'):
            app_context.llm_service_instance.logger.warning(
                f"OCR was disabled via web interface. Switched to image format: {next_format}"
            )
    
    return jsonify({'success': True, 'enabled': enabled, 'image_format': app_context.global_settings.get("VLM_IMAGE_FORMAT", "markdown")})

@app.route('/api/intervals', methods=['POST'])
def update_intervals():
    if not app_context:
        return jsonify({'error': 'Server not initialized'}), 500
    
    data = request.get_json()
    min_interval = float(data.get('min_interval', 30.0))
    max_interval = float(data.get('max_interval', 120.0))
    
    if min_interval >= max_interval:
        return jsonify({'error': 'Minimum interval must be less than maximum interval'}), 400
    
    # Update the global settings
    app_context.global_settings["NDI_MIN_RANDOM_COMMENTARY_INTERVAL_S"] = min_interval
    app_context.global_settings["NDI_MAX_RANDOM_COMMENTARY_INTERVAL_S"] = max_interval
    
    # Force recalculation of next commentary delay in LLM service
    if app_context.llm_service_instance:
        app_context.llm_service_instance.next_vlm_commentary_delay = \
            app_context.llm_service_instance._calculate_next_commentary_delay()
    
    return jsonify({'success': True})

@app.route('/api/chat', methods=['POST'])
def chat():
    if not app_context:
        return jsonify({'error': 'Server not initialized'}), 500
    
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'error': 'Message cannot be empty'}), 400
    
    try:
        # Set conversation flag
        app_context.is_in_conversation.set()
        
        # Use the LLM service to handle the chat
        app_context.llm_service_instance.handle_user_text_query(message, "WebUI")
        
        # Since handle_user_text_query is async (puts response in queue),
        # we need to get the response from the text_message_queue
        try:
            response = app_context.text_message_queue.get(timeout=30)  # 30 second timeout
            app_context.text_message_queue.task_done()
            
            # Clean up the response
            response = clean_text_for_tts(response)
            
        except Exception as e:
            response = "Sorry, I didn't receive a response in time."
        
        return jsonify({'success': True, 'response': response})
        
    except Exception as e:
        app_context.logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/image-format', methods=['POST'])
def set_image_format():
    if not app_context:
        return jsonify({'error': 'Server not initialized'}), 500
    
    data = request.get_json()
    format_name = data.get('format')
    
    valid_formats = ['markdown', 'llava', 'html', 'cogvlm', 'qwen', 'raw']
    
    if not format_name or format_name not in valid_formats:
        return jsonify({'error': f'Invalid format. Must be one of: {", ".join(valid_formats)}'}), 400
    
    # Update the image format in global settings
    app_context.global_settings["VLM_IMAGE_FORMAT"] = format_name
    app_context.logger.info(f"Image format updated to '{format_name}' via web interface")
    
    # Log in LLM service as well
    if hasattr(app_context.llm_service_instance, 'logger'):
        app_context.llm_service_instance.logger.info(
            f"Image format changed to '{format_name}' via web interface"
        )
    
    return jsonify({'success': True, 'format': format_name})

@app.route('/api/capture-screenshot', methods=['POST'])
def capture_screenshot():
    """Endpoint for manually capturing a screenshot and sending to LLM"""
    if not app_context:
        return jsonify({'error': 'Server not initialized'}), 500
    
    # Get image size from request if provided
    data = request.get_json() or {}
    max_image_size = int(data.get('max_size', 512))
    
    # Update global settings with the max image size
    app_context.global_settings["VLM_MAX_IMAGE_SIZE"] = max_image_size
    app_context.logger.info(f"Setting max image size to {max_image_size}px")
    
    # Check if NDI service is running and can provide frames
    if not app_context.ndi_service_instance or not hasattr(app_context.ndi_service_instance, 'last_captured_frame'):
        return jsonify({'error': 'NDI service not initialized or no frames available'}), 500
    
    # Get the last captured frame
    last_frame = getattr(app_context.ndi_service_instance, 'last_captured_frame', None)
    if last_frame is None:
        return jsonify({'error': 'No frames have been captured yet'}), 404
    
    # Check if we have an LLM service
    if not app_context.llm_service_instance:
        return jsonify({'error': 'LLM service not initialized'}), 500
    
    # Force manual frame analysis, bypassing commentary delay
    app_context.logger.info("Manual screenshot capture requested. Sending to VLM for analysis.")
    
    try:
        # Store the original delay
        original_delay = app_context.llm_service_instance.next_vlm_commentary_delay
        
        # Set delay to 0 to force immediate processing
        app_context.llm_service_instance.next_vlm_commentary_delay = 0
        app_context.llm_service_instance.last_vlm_time = 0
        
        # Process the frame
        app_context.llm_service_instance.generate_vlm_commentary_from_frame(last_frame)
        
        # Restore the original delay
        app_context.llm_service_instance.next_vlm_commentary_delay = original_delay
        
        return jsonify({
            'success': True, 
            'message': 'Screenshot captured and sent for analysis',
            'image_format': app_context.global_settings.get("VLM_IMAGE_FORMAT", "qwen"),
            'image_size': max_image_size
        })
        
    except Exception as e:
        app_context.logger.error(f"Error during manual screenshot capture: {e}", exc_info=True)
        return jsonify({'error': f'Error processing screenshot: {str(e)}'}), 500

@app.route('/api/image-size', methods=['POST'])
def set_image_size():
    """Endpoint for setting maximum image size"""
    if not app_context:
        return jsonify({'error': 'Server not initialized'}), 500
    
    data = request.get_json()
    max_size = data.get('max_size')
    
    try:
        max_size = int(max_size)
        if max_size < 64 or max_size > 2048:
            return jsonify({'error': 'Image size must be between 64 and 2048 pixels'}), 400
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid image size value'}), 400
    
    # Update the image size in global settings
    app_context.global_settings["VLM_MAX_IMAGE_SIZE"] = max_size
    app_context.logger.info(f"Image size updated to {max_size}px via web interface")
    
    # Log in LLM service as well
    if hasattr(app_context.llm_service_instance, 'logger'):
        app_context.llm_service_instance.logger.info(
            f"Max image size changed to {max_size}px via web interface"
        )
    
    return jsonify({'success': True, 'max_size': max_size})

@app.route('/api/prompts', methods=['GET'])
def get_prompts():
    """Endpoint to get all current prompts"""
    if not app_context:
        return jsonify({'error': 'Server not initialized'}), 500
    
    # Get active profile
    profile = app_context.active_profile
    
    # Extract prompts from profile
    prompts = {
        'system_prompt_commentary': getattr(profile, 'system_prompt_commentary', 'No system prompt defined'),
        'user_prompt_template_commentary_ocr': getattr(profile, 'user_prompt_template_commentary', 'No OCR enabled prompt defined'),
        'ocr_disabled_template': app_context.global_settings.get("OCR_DISABLED_PROMPT_TEMPLATE", 
            "Analyze the visual content of the game using the full screenshot provided. "
            "Focus on overall scene composition, character actions, environment changes, and notable events. "
            "Game: {game_name}. What interesting observations can you make about what you can see in the image?"),
        'system_prompt_chat': getattr(profile, 'system_prompt_chat', 'No chat system prompt defined')
    }
    
    return jsonify(prompts)

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    """Endpoint for uploading an image file for VLM analysis"""
    if not app_context:
        return jsonify({'error': 'Server not initialized'}), 500
    
    # Check if we have an LLM service
    if not app_context.llm_service_instance:
        return jsonify({'error': 'LLM service not initialized'}), 500
        
    # Check if the post has a file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    # Get image size from request if provided
    max_image_size = int(request.form.get('max_size', 512))
    
    # Update global settings with the max image size
    app_context.global_settings["VLM_MAX_IMAGE_SIZE"] = max_image_size
    app_context.logger.info(f"Setting max image size to {max_image_size}px for uploaded image")
    
    try:
        # Read the file into a numpy array
        img_stream = file.read()
        img_array = np.frombuffer(img_stream, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Could not decode uploaded image'}), 400
            
        app_context.logger.info(f"Successfully decoded uploaded image: {frame.shape}")
        
        # Set debug mode for maximum logging
        app_context.global_settings["VLM_DEBUG_MODE"] = True
        app_context.logger.info("Setting VLM_DEBUG_MODE=True for uploaded image analysis")
        
        # Store the original delay
        original_delay = app_context.llm_service_instance.next_vlm_commentary_delay
        
        # Set delay to 0 to force immediate processing
        app_context.llm_service_instance.next_vlm_commentary_delay = 0
        app_context.llm_service_instance.last_vlm_time = 0
        
        # Process the frame
        app_context.llm_service_instance.generate_vlm_commentary_from_frame(frame)
        
        # Restore the original delay
        app_context.llm_service_instance.next_vlm_commentary_delay = original_delay
        
        return jsonify({
            'success': True, 
            'message': 'Image uploaded and sent for analysis',
            'image_format': app_context.global_settings.get("VLM_IMAGE_FORMAT", "qwen"),
            'image_size': max_image_size,
            'shape': f"{frame.shape[1]}x{frame.shape[0]}"
        })
        
    except Exception as e:
        app_context.logger.error(f"Error processing uploaded image: {e}", exc_info=True)
        return jsonify({'error': f'Error processing uploaded image: {str(e)}'}), 500

@app.route('/api/local-image', methods=['POST'])
def analyze_local_image():
    """Endpoint for analyzing an image from a local path on the server"""
    if not app_context:
        return jsonify({'error': 'Server not initialized'}), 500
        
    # Check if we have an LLM service
    if not app_context.llm_service_instance:
        return jsonify({'error': 'LLM service not initialized'}), 500
        
    data = request.get_json()
    image_path = data.get('path', '').strip()
    
    if not image_path:
        return jsonify({'error': 'No image path provided'}), 400
        
    # Get image size from request if provided
    max_image_size = int(data.get('max_size', 512))
    
    # Update global settings with the max image size
    app_context.global_settings["VLM_MAX_IMAGE_SIZE"] = max_image_size
    app_context.logger.info(f"Setting max image size to {max_image_size}px for local image")
    
    try:
        # Check if file exists
        if not os.path.isfile(image_path):
            return jsonify({'error': f'File not found: {image_path}'}), 404
            
        # Read the image
        frame = cv2.imread(image_path)
        
        if frame is None:
            return jsonify({'error': f'Could not read image at path: {image_path}'}), 400
            
        app_context.logger.info(f"Successfully read local image from {image_path}: {frame.shape}")
        
        # Set debug mode for maximum logging
        app_context.global_settings["VLM_DEBUG_MODE"] = True
        app_context.logger.info("Setting VLM_DEBUG_MODE=True for local image analysis")
        
        # Store the original delay
        original_delay = app_context.llm_service_instance.next_vlm_commentary_delay
        
        # Set delay to 0 to force immediate processing
        app_context.llm_service_instance.next_vlm_commentary_delay = 0
        app_context.llm_service_instance.last_vlm_time = 0
        
        # Process the frame
        app_context.llm_service_instance.generate_vlm_commentary_from_frame(frame)
        
        # Restore the original delay
        app_context.llm_service_instance.next_vlm_commentary_delay = original_delay
        
        return jsonify({
            'success': True, 
            'message': 'Local image sent for analysis',
            'path': image_path,
            'image_format': app_context.global_settings.get("VLM_IMAGE_FORMAT", "qwen"),
            'image_size': max_image_size,
            'shape': f"{frame.shape[1]}x{frame.shape[0]}"
        })
        
    except Exception as e:
        app_context.logger.error(f"Error processing local image: {e}", exc_info=True)
        return jsonify({'error': f'Error processing local image: {str(e)}'}), 500

@app.route('/api/run-diagnostic', methods=['POST'])
def run_deep_diagnostic():
    """Endpoint for triggering a deep diagnostic test of all image formats"""
    if not app_context:
        return jsonify({'error': 'Server not initialized'}), 500
        
    # Check if we have an LLM service
    if not app_context.llm_service_instance:
        return jsonify({'error': 'LLM service not initialized'}), 500
    
    data = request.get_json() or {}
    
    # Check if we have an image source - either an uploaded file, a path, or use the latest frame
    image_source = data.get('source', 'latest')  # 'latest', 'upload', or 'path'
    image_path = data.get('path', '')
    max_image_size = int(data.get('max_size', 512))
    
    # Update global settings
    app_context.global_settings["VLM_MAX_IMAGE_SIZE"] = max_image_size
    app_context.global_settings["VLM_DEBUG_MODE"] = True
    app_context.global_settings["VLM_DEEP_DIAGNOSTIC"] = True
    
    app_context.logger.info(f"Initiating deep diagnostic mode with source: {image_source}, max size: {max_image_size}px")
    
    # Get the image frame based on the source
    frame = None
    
    try:
        if image_source == 'path' and image_path:
            # Check if file exists
            if not os.path.isfile(image_path):
                return jsonify({'error': f'File not found: {image_path}'}), 404
                
            # Read image from local path
            frame = cv2.imread(image_path)
            if frame is None:
                return jsonify({'error': f'Could not read image at path: {image_path}'}), 400
                
            app_context.logger.info(f"Using local image for diagnostic: {image_path}, shape: {frame.shape}")
            
        elif image_source == 'latest':
            # Use the latest captured frame
            if not app_context.ndi_service_instance or not hasattr(app_context.ndi_service_instance, 'last_captured_frame'):
                return jsonify({'error': 'NDI service not initialized or no frames available'}), 500
                
            frame = getattr(app_context.ndi_service_instance, 'last_captured_frame', None)
            if frame is None:
                return jsonify({'error': 'No frames have been captured yet'}), 404
                
            app_context.logger.info(f"Using latest captured frame for diagnostic, shape: {frame.shape}")
        
        else:
            return jsonify({'error': f'Invalid image source: {image_source}'}), 400
        
        # Store the original delay
        original_delay = app_context.llm_service_instance.next_vlm_commentary_delay
        
        # Set delay to 0 to force immediate processing 
        app_context.llm_service_instance.next_vlm_commentary_delay = 0
        app_context.llm_service_instance.last_vlm_time = 0
        
        # Process the frame - the VLM_DEEP_DIAGNOSTIC flag will trigger the diagnostic mode
        app_context.llm_service_instance.generate_vlm_commentary_from_frame(frame)
        
        # Restore the original delay
        app_context.llm_service_instance.next_vlm_commentary_delay = original_delay
        
        # Reset diagnostic mode after running
        app_context.global_settings["VLM_DEEP_DIAGNOSTIC"] = False
        
        return jsonify({
            'success': True,
            'message': 'Deep diagnostic test started. Results will be saved to debug_vlm_frames directory and sent to chat.',
            'image_size': max_image_size,
            'source': image_source
        })
        
    except Exception as e:
        # Reset diagnostic mode after error
        app_context.global_settings["VLM_DEEP_DIAGNOSTIC"] = False
        app_context.logger.error(f"Error running deep diagnostic: {e}", exc_info=True)
        return jsonify({'error': f'Error running deep diagnostic: {str(e)}'}), 500

def run_server(app_ctx, host='0.0.0.0', port=5000):
    """
    Run the Flask server in a separate thread.
    Using 0.0.0.0 to make it accessible from Windows host.
    """
    global app_context
    app_context = app_ctx
    
    # Create required directories
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static', 'css'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static', 'js'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
    
    def _run():
        app.run(host=host, port=port, debug=False, use_reloader=False)
    
    server_thread = threading.Thread(target=_run, daemon=True)
    server_thread.start()
    
    app_context.logger.info(f"Web interface available at http://{host}:{port}")
    return server_thread 