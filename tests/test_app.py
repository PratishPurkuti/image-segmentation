import unittest
import os
import shutil
import tempfile
import io
import base64
from PIL import Image, ImageDraw
import sys
import unittest.mock

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from utils import segmentation

class TestInstanceSegmentationApp(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
        self.client = app.test_client()
        
    def tearDown(self):
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        
    def create_test_image(self):
        # Create a simple image with a red circle and blue square
        img = Image.new('RGB', (200, 200), color='white')
        draw = ImageDraw.Draw(img)
        draw.ellipse((50, 50, 150, 150), fill='red')
        draw.rectangle((10, 10, 40, 40), fill='blue')
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr
        
    def test_upload_no_file(self):
        response = self.client.post('/upload')
        self.assertEqual(response.status_code, 400)
        
    def test_upload_invalid_file(self):
        data = {'file': (io.BytesIO(b'fake'), 'test.txt')}
        response = self.client.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)

    # Note: We cannot easily test the full external API call in unit tests without a mock.
    # So we'll mock the segment_image function.
    def test_processing_flow(self):
        # Mock segmentation result
        # Create a fake mask (base64)
        mask_img = Image.new('L', (200, 200), 0)
        draw = ImageDraw.Draw(mask_img)
        draw.ellipse((50, 50, 150, 150), fill=255) # Match the red circle
        
        mask_byte_arr = io.BytesIO()
        mask_img.save(mask_byte_arr, format='PNG')
        mask_b64 = base64.b64encode(mask_byte_arr.getvalue()).decode('utf-8')
        
        mock_response = [
            {'label': 'test_object', 'score': 0.9, 'mask': mask_b64}
        ]
        
        # Patch specifically where it is looked up in app.py
        # Since app.py does "from utils.segmentation import segment_image"
        # we must patch "app.segment_image"
        with unittest.mock.patch('app.segment_image', return_value=mock_response):
            img_data = self.create_test_image()
            data = {'file': (img_data, 'test.png')}
            
            response = self.client.post('/upload', data=data, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 200)
            json_data = response.get_json()
            
            self.assertIn('session_id', json_data)
            self.assertIn('files', json_data)
            self.assertTrue(len(json_data['files']) > 0)
            self.assertIn('zip_file', json_data)

if __name__ == '__main__':
    unittest.main()
