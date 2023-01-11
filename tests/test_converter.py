import unittest
from usd2gltf.converter import convert


class TestConverter(unittest.TestCase):

	def test_converter(self):
		self.assertEqual(convert("usd"), "converted usd to gltf")

if __name__ == '__main__':
    unittest.main()
