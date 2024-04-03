import unittest
import source

class TestAddition(unittest.TestCase):
    def testAddition1(self): 
        self.assertEqual(3, source.performAdd(2, 1), "Error in implementation.")

    def testAddition2(self):
        self.assertEqual(30, source.performAdd(20, 10), "Error in implementation.")

class TestSub(unittest.TestCase):
    def testSub1(self):
        self.assertEqual(3, source.performSub(6, 3), "Error in implementation.")

class TestMult(unittest.TestCase):
    def testMult1(self):
        self.assertEqual(5, source.performMult(5, 1), "Error in implementation.")
    def testMult2(self):
        self.assertEqual(0, source.performMult(5, 0), "Error in implementation.")

class TestDiv(unittest.TestCase):
    def testDiv1(self):
        self.assertEqual(2, source.performDiv(10, 5), "Error in implementation.")
    def testDiv2(self):
        self.assertEqual("Error: Cannot Divide by Zero", source.performDiv(10, 0), "Error in implementation.")

if __name__ == '__main__': 
    unittest.main() 