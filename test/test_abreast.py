import os, unittest
import numpy as np
import abreast

class TestAbreast(unittest.TestCase):

    def test_generate(self):
        my_shape = abreast.Abreast()
        self.assertIsInstance(my_shape, abreast.Abreast)
        my_shape.generate()
        self.assertIsInstance(my_shape.generated_shape, np.ndarray)
        my_shape.generate(45)
        self.assertLessEqual(np.abs(my_shape.generated_shape[my_shape._cc_angles,my_shape._cc_slices-1]-44.5), 0.5)
        my_shape.generate(thickness = 40, gauss_approx=False)

    def test_export_obj(self):
        test_name = "test"
        test_name_obj = "test.obj"
        my_shape = abreast.Abreast()
        my_shape.generate(thickness = 50)
        my_shape.export(test_name)
        self.assertTrue(os.path.exists(test_name_obj))
        os.remove(test_name_obj)

    def test_export_volume(self):
        test_name = "test"
        test_name_vox = "test.tif"
        my_shape = abreast.Abreast()
        my_shape.generate(thickness = 50)
        my_shape.export(test_name, format="vox")
        self.assertTrue(os.path.exists(test_name_vox))
        os.remove(test_name_vox)

    def test_export_slices(self):
        test_name = "test_dir"
        my_shape = abreast.Abreast()
        my_shape.generate(thickness = 50)
        my_shape.export(test_name, format="vox", exportSlices=True)
        self.assertTrue(os.path.exists(test_name))
        exported_slices = os.listdir(test_name)
        self.assertTrue(len(exported_slices), 50)
        for slice in exported_slices:
            os.remove(os.path.join(os.curdir, test_name, slice))
        os.rmdir(test_name)

if __name__ == '__main__':
    unittest.main()
