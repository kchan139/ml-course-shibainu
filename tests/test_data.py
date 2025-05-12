import os
import pandas as pd
import unittest
from tempfile import TemporaryDirectory
from src.data.utils import remove_columns


class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        
        self.test_data = pd.DataFrame({
            'Id': [1, 2],
            'ProductId': ['B001E4KFG0', 'B00813GRG4'],
            'UserId': ['A3SGXH7AUHU8GW', 'A1D87F6ZCVE5NK'],
            'ProfileName': ['delmartian', 'dll pa'],
            'HelpfulnessNumerator': [1, 0],
            'HelpfulnessDenominator': [1, 0],
            'Score': [5, 1],
            'Time': [1303862400, 1346976000],
            'Summary': ['Good Quality Dog Food', 'Not as Advertised'],
            'Text': ['Sample text 1', 'Sample text 2']
        })
        
        self.input_file = os.path.join(self.temp_dir.name, 'input.csv')
        self.test_data.to_csv(self.input_file, index=False)
        
        self.output_file = os.path.join(self.temp_dir.name, 'processed', 'output.csv')
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_remove_columns(self):
        remove_columns(self.input_file, self.output_file)
        
        self.assertTrue(os.path.exists(self.output_file))
        
        df_processed = pd.read_csv(self.output_file)
        
        removed_columns = ['ProductId', 'UserId', 'ProfileName', 
                         'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time']
        for col in removed_columns:
            self.assertNotIn(col, df_processed.columns)
        
        preserved_columns = ['Id', 'Score', 'Summary', 'Text']
        for col in preserved_columns:
            self.assertIn(col, df_processed.columns)
        
        self.assertEqual(len(df_processed), 2)
        self.assertEqual(df_processed['Id'].tolist(), [1, 2])
        self.assertEqual(df_processed['Score'].tolist(), [5, 1])
    
    def test_missing_columns(self):
        partial_data = self.test_data.drop(columns=['UserId', 'Time'])
        partial_file = os.path.join(self.temp_dir.name, 'partial.csv')
        partial_data.to_csv(partial_file, index=False)
        
        output_file = os.path.join(self.temp_dir.name, 'processed', 'partial_output.csv')
        
        remove_columns(partial_file, output_file)
        
        df_processed = pd.read_csv(output_file)
        
        columns_should_be_removed = ['ProductId', 'ProfileName', 
                                   'HelpfulnessNumerator', 'HelpfulnessDenominator']
        for col in columns_should_be_removed:
            self.assertNotIn(col, df_processed.columns)
        
        self.assertIn('Id', df_processed.columns)
        self.assertIn('Score', df_processed.columns)


if __name__ == '__main__':
    unittest.main()