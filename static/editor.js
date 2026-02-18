class ImageEditor {
    constructor(canvasId, resultCallback) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.lastX = 0;
        this.lastY = 0;
        this.brushSize = 20;
        this.image = null;
        this.maskCanvas = document.createElement('canvas'); // For storing strokes
        this.maskCtx = this.maskCanvas.getContext('2d');
        this.scale = 1;

        // Bind events
        this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
        this.canvas.addEventListener('mousemove', this.draw.bind(this));
        this.canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
        this.canvas.addEventListener('mouseout', this.stopDrawing.bind(this));

        // Touch support
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent("mousedown", {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            this.canvas.dispatchEvent(mouseEvent);
        }, { passive: false });

        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent("mousemove", {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            this.canvas.dispatchEvent(mouseEvent);
        }, { passive: false });

        this.canvas.addEventListener('touchend', (e) => {
            const mouseEvent = new MouseEvent("mouseup", {});
            this.canvas.dispatchEvent(mouseEvent);
        });

        this.history = [];
        this.redoStack = [];
        this.maxHistory = 20;
    }

    loadImage(src) {
        this.image = new Image();
        this.image.crossOrigin = "Anonymous";
        this.image.onload = () => {
            // Resize canvas to fit in view while maintaining aspect ratio
            // Actual image dimensions
            this.canvas.width = this.image.width;
            this.canvas.height = this.image.height;

            // Mask canvas (same size)
            this.maskCanvas.width = this.image.width;
            this.maskCanvas.height = this.image.height;
            this.maskCtx.lineCap = 'round';
            this.maskCtx.lineJoin = 'round';
            this.maskCtx.strokeStyle = 'white'; // White = erase
            this.maskCtx.lineWidth = this.brushSize;

            // Reset history
            this.history = [];
            this.redoStack = [];

            this.render();
        };
        this.image.src = src;
    }

    setBrushSize(size) {
        this.brushSize = size;
        this.maskCtx.lineWidth = size;
    }

    saveState() {
        // Save current mask state to history
        const imageData = this.maskCtx.getImageData(0, 0, this.maskCanvas.width, this.maskCanvas.height);
        this.history.push(imageData);

        // Cap history size
        if (this.history.length > this.maxHistory) {
            this.history.shift();
        }

        // Clear redo stack on new action
        this.redoStack = [];
    }

    startDrawing(e) {
        this.isDrawing = true;
        this.saveState(); // Save state BEFORE drawing
        [this.lastX, this.lastY] = this.getCoordinates(e);
    }

    draw(e) {
        if (!this.isDrawing) return;
        const [x, y] = this.getCoordinates(e);

        // Draw on mask canvas (invisible logic layer)
        this.maskCtx.beginPath();
        this.maskCtx.moveTo(this.lastX, this.lastY);
        this.maskCtx.lineTo(x, y);
        this.maskCtx.stroke();

        [this.lastX, this.lastY] = [x, y];

        // Re-render visible canvas
        this.render();
    }

    stopDrawing() {
        this.isDrawing = false;
    }

    getCoordinates(e) {
        // Get mouse position relative to canvas
        const rect = this.canvas.getBoundingClientRect();

        // Calculate scale (CSS size vs Actual size)
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        return [
            (e.clientX - rect.left) * scaleX,
            (e.clientY - rect.top) * scaleY
        ];
    }

    render() {
        // 1. Draw original image
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.image, 0, 0);

        // 2. Draw mask overlay (Red semi-transparent)
        // We can draw the maskCanvas on top with a red tint

        // Save context
        this.ctx.save();

        // Draw the strokes (maskCanvas) in red to show what will be erased
        // We can do this by drawing the maskCanvas as a stencil
        this.ctx.globalAlpha = 0.5;
        this.ctx.globalCompositeOperation = 'source-over';

        // Create a temporary red canvas from mask
        const redCanvas = document.createElement('canvas');
        redCanvas.width = this.canvas.width;
        redCanvas.height = this.canvas.height;
        const redCtx = redCanvas.getContext('2d');
        redCtx.fillStyle = 'red';
        redCtx.fillRect(0, 0, redCanvas.width, redCanvas.height);

        // Use mask to cut out
        redCtx.globalCompositeOperation = 'destination-in';
        redCtx.drawImage(this.maskCanvas, 0, 0);

        // Draw red stroke on main canvas
        this.ctx.drawImage(redCanvas, 0, 0);

        this.ctx.restore();
    }

    undo() {
        if (this.history.length === 0) return;

        // Save current state to redo stack
        const currentData = this.maskCtx.getImageData(0, 0, this.maskCanvas.width, this.maskCanvas.height);
        this.redoStack.push(currentData);

        // Restore last state
        const lastData = this.history.pop();
        this.maskCtx.putImageData(lastData, 0, 0);

        this.render();
    }

    redo() {
        if (this.redoStack.length === 0) return;

        // Save current state to history
        const currentData = this.maskCtx.getImageData(0, 0, this.maskCanvas.width, this.maskCanvas.height);
        this.history.push(currentData);

        // Restore redo state
        const nextData = this.redoStack.pop();
        this.maskCtx.putImageData(nextData, 0, 0);

        this.render();
    }

    getMask() {
        return this.maskCanvas.toDataURL('image/png');
    }
}
