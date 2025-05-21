import os
import unittest
from tempfile import TemporaryDirectory
import pandas as pd
from src.data.utils import *


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

    def test_html_cleaning(self):
        html_data = pd.DataFrame({
            'Id': [1, 2, 3],
            'Score': [5, 4, 3],
            'Summary': ['Test 1', 'Test 2', 'Test 3'],
            'Text': [
                '<p>Simple paragraph</p>',
                'Text without HTML',
                '<div class="complex">Nested <b>bold</b> text</div>'
            ]
        })
        
        html_file = os.path.join(self.temp_dir.name, 'html_test.csv')
        html_data.to_csv(html_file, index=False)
        
        output_file = os.path.join(self.temp_dir.name, 'processed', 'html_cleaned.csv')
        
        process_html_tags(html_file, output_file)
        processed_df = pd.read_csv(output_file)
        
        self.assertEqual(processed_df['Text_HTML_Clean'][0], 'Simple paragraph')
        self.assertEqual(processed_df['Text_HTML_Clean'][1], 'Text without HTML')
        self.assertEqual(processed_df['Text_HTML_Clean'][2], 'Nested bold text')
        
        self.assertEqual(processed_df['Text'][0], '<p>Simple paragraph</p>')

    def test_text_length_normalization(self):
        # Create test data with varying text lengths
        length_data = pd.DataFrame({
            'Id': [1, 2, 3],
            'Score': [5, 4, 3],
            'Summary': ['Test 1', 'Test 2', 'Test 3'],
            'Text_HTML_Clean': [
                'Short text',
                'This is a medium length text that should be within normal limits',
                'This is an extremely long text ' * 50  # Very long text
            ]
        })
        
        length_file = os.path.join(self.temp_dir.name, 'length_test.csv')
        length_data.to_csv(length_file, index=False)
        
        output_file = os.path.join(self.temp_dir.name, 'processed', 'length_normalized.csv')
        
        # Process the data with length normalization
        normalize_text_length(length_file, output_file, p95_threshold=100)
        
        # Load the processed data
        processed_df = pd.read_csv(output_file)
        
        # Check truncation worked
        self.assertEqual(processed_df['Text_Normalized'][0], 'Short text')  # Short text unchanged
        self.assertTrue(len(processed_df['Text_Normalized'][1]) <= 100)  # Medium text under threshold
        self.assertEqual(len(processed_df['Text_Normalized'][2]), 100)  # Long text truncated
        
        # Check length features are calculated
        self.assertTrue('Length_Feature' in processed_df.columns)
        self.assertTrue('Length_Category' in processed_df.columns)

    def test_length_categories(self):
        # Test data with specific lengths for each category
        category_data = pd.DataFrame({
            'Id': range(1, 8),
            'Score': [5] * 7,
            'Text_HTML_Clean': [
                'a' * 50,     # Very Short (0-100)
                'a' * 150,    # Short (100-250)
                'a' * 350,    # Medium-Short (250-500)
                'a' * 600,    # Medium (500-750)
                'a' * 850,    # Medium-Long (750-1000)
                'a' * 1200,   # Long (1000-1500)
                'a' * 2000,   # Very Long (>1500)
            ]
        })
        
        category_file = os.path.join(self.temp_dir.name, 'category_test.csv')
        category_data.to_csv(category_file, index=False)
        
        output_file = os.path.join(self.temp_dir.name, 'processed', 'categorized.csv')
        
        # Process the data with length categorization
        categorize_text_length(category_file, output_file)
        
        # Load the processed data
        processed_df = pd.read_csv(output_file)
        
        # Check correct categories assigned
        expected_categories = [
            'Very Short', 'Short', 'Medium-Short', 'Medium', 
            'Medium-Long', 'Long', 'Very Long'
        ]
        
        for i, expected in enumerate(expected_categories):
            self.assertEqual(processed_df['Length_Category'][i], expected)


class TestCleaningFunctions(unittest.TestCase):
    def test_clean_html_function(self):
        test_cases = [
            ("<p>Simple paragraph</p>", "Simple paragraph"),
            ('<div class="complex">Nested <b>bold</b> text</div>', "Nested bold text"),
            ("<ul><li>Item 1</li><li>Item 2</li></ul>", "Item 1 Item 2"),
            ("No HTML here", "No HTML here"),
            ("", ""),
        ]
        
        for html, expected in test_cases:
            with self.subTest(html=html):
                self.assertEqual(clean_html(html), expected)
    
    def test_contains_html_tags_function(self):
        test_cases = [
            ("<p>Has tags</p>", True),
            ("No tags here", False),
            ("<incomplete tag", False),
            ("Text with > and < symbols", False),
            ("", False),
            (None, False)
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                self.assertEqual(contains_html_tags(text), expected)


if __name__ == '__main__':
    unittest.main()