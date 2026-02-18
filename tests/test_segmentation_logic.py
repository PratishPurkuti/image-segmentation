import unittest
import os
import shutil
import base64
import io
import numpy as np
from PIL import Image
from utils.segmentation import extract_objects

class TestSegmentationLogic(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_output"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a dummy image (100x100 red square)
        self.image_path = os.path.join(self.test_dir, "test_image.png")
        img = Image.new("RGB", (100, 100), "red")
        img.save(self.image_path)
        
        # Create a dummy mask (50x50 white square in center, rest black)
        # We simulate the API return which might be a smaller mask or base64
        mask_img = Image.new("L", (50, 50), 0)
        # Draw a white circle or square
        mask_arr = np.array(mask_img)
        mask_arr[10:40, 10:40] = 255
        mask_img = Image.fromarray(mask_arr)
        
        # Encode to base64 to simulate API response
        img_byte_arr = io.BytesIO()
        mask_img.save(img_byte_arr, format='PNG')
        self.mask_b64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_extract_objects(self):
        # Simulate API results
        results = [{
            'label': 'test_obj',
            'score': 0.99,
            'mask': self.mask_b64
        }]
        
        extracted = extract_objects(self.image_path, results, self.test_dir)
        
        self.assertTrue(len(extracted) > 0)
        self.assertTrue(os.path.exists(extracted[0]))
        
        # Verify the output image properties
        out_img = Image.open(extracted[0])
        self.assertEqual(out_img.mode, "RGBA")
        # Should be roughly the size of the mask content (stretched to 100x100 relative)
        # The mask was 50x50, with 30x30 white box.
        # Scaled up to 100x100, the white box should be roughly 60x60.
        print(f"Output image size: {out_img.size}")

if __name__ == '__main__':
    unittest.main()
