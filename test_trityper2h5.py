import os
import unittest
import trityper2h5
from tempfile import TemporaryDirectory


class TriTyperDataTest(unittest.TestCase):
    def test_trityper_data(self):
        trityper_data = trityper2h5.TriTyperData("exampleTriTyper")
        self.assertEqual(trityper_data.number_of_variants, 199)
        self.assertEqual(len(trityper_data.individuals_data), 500)


class TriTyper2h5Test(unittest.TestCase):
    def test_trityper2h5(self):
        with TemporaryDirectory() as temporary_directory:

            self.assertFalse(trityper2h5.main(
                "--input exampleTriTyper --output {} --study_name dosage"
                    .format(os.path.join(temporary_directory, "haseh5")).split(" ")))


if __name__ == '__main__':
    unittest.main()
