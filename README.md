# Instance Segmentation Web App

A Flask-based web application that automatically extracts objects from images using AI. It uses the Hugging Face Inference API (Mask2Former) to perform instance segmentation, cropping each detected object and saving it as a transparent PNG.

![App Screenshot](https://via.placeholder.com/800x400?text=Instance+Segmentation+App+Preview)

## Features

- **Drag & Drop Upload**: Easy to use interface.
- **AI-Powered**: Uses state-of-the-art instance segmentation models.
- **Transparent Exports**: Automatically removes backgrounds from extracted objects.
- **Batch Download**: Download individual objects or a ZIP file of all extractions.
- **Secure & Private**: Files are processed in memory/temp storage and automatically cleaned up.

## Prerequisites

- Python 3.8 or higher
- A Hugging Face Account (Free)

## Installation

1. **Clone the repository** (or download the source code):
   ```bash
   git clone https://github.com/yourusername/instance-segmentation-app.git
   cd instance-segmentation-app
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## API Configuration

This app uses the Hugging Face Inference API. You need a free API token to run it.

1.  **Get your Token**:
    *   Go to [Hugging Face Settings > Tokens](https://huggingface.co/settings/tokens).
    *   Click **"Create new token"**.
    *   Name it `InstanceSegmenter` (or anything you like).
    *   Select **"Read"** permissions (default).
    *   Click **"Create token"** and copy the string (starts with `hf_`).

2.  **Configure the App**:
    *   Create a file named `.env` in the project root directory.
    *   Add your token to it:
    
    ```env
    HF_API_TOKEN=hf_your_generated_token_here
    ```

## Usage

1.  **Start the Application**:
    ```bash
    python app.py
    ```

2.  **Open in Browser**:
    *   Navigate to `http://127.0.0.1:5000`

3.  **Extract Objects**:
    *   Drag and drop an image (JPG/PNG).
    *   Wait for the AI to process.
    *   Download extracted objects individually or as a ZIP.

## Troubleshooting

- **401 Unauthorized**: Check your `.env` file and ensure the `HF_API_TOKEN` is correct.
- **503 Service Unavailable**: The model might be loading (cold boot). Wait a moment and try again; the app automatically sets `x-wait-for-model: true`.

## License

MIT License. Feel free to use and modify!
