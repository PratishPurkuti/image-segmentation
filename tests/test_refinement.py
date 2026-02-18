import unittest
import os
import shutil
import base64
import io
import numpy as np
from PIL import Image
from app import app

class TestRefinement(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.upload_folder = app.config['UPLOAD_FOLDER']
        self.session_id = "test_refinement_session"
        self.session_dir = os.path.join(self.upload_folder, 'instance_seg_app', self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Create a dummy image (100x100 solid red, fully opaque)
        self.filename = "test_obj.png"
        self.file_path = os.path.join(self.session_dir, self.filename)
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        img.save(self.file_path)

    def tearDown(self):
        if os.path.exists(self.session_dir):
            shutil.rmtree(self.session_dir)

    def test_refine_endpoint(self):
        # Create a mask (100x100), with a 50x50 white square in center (which should be erased)
        mask_img = Image.new("L", (100, 100), 0)
        mask_arr = np.array(mask_img)
        mask_arr[25:75, 25:75] = 255 # White square in middle
        mask_img = Image.fromarray(mask_arr)
        
        # Save to base64
        img_byte_arr = io.BytesIO()
        mask_img.save(img_byte_arr, format='PNG')
        mask_b64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        # Call /refine
        response = self.app.post('/refine', json={
            'session_id': self.session_id,
            'filename': self.filename,
            'mask': mask_b64
        })
        
        self.assertEqual(response.status_code, 200)
        
        # Verify the image was modified
        modified_img = Image.open(self.file_path).convert("RGBA")
        r, g, b, a = modified_img.split()
        a_arr = np.array(a)
        
        # The center 50x50 should now be transparent (alpha=0)
        center_alpha = a_arr[50, 50]
        self.assertEqual(center_alpha, 0, "Center pixel should be transparent")
        
        # The corner should still be opaque
        corner_alpha = a_arr[0, 0]
        self.assertEqual(corner_alpha, 255, "Corner pixel should be opaque")

if __name__ == '__main__':
    unittest.main()
