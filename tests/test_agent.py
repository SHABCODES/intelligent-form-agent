import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.main import IntelligentFormAgent

class TestIntelligentFormAgent(unittest.TestCase):
    
    def setUp(self):
        self.agent = IntelligentFormAgent()
        
        self.structured_invoice = """
        TAX INVOICE
        Invoice Number: INV/2023/001
        Date: 15-10-2023
        Customer: Sharma Enterprises
        Email: rajesh.sharma@sharmaenterprises.com
        Phone: +91 98765 43210
        Total Amount: ₹1,53,400.00
        """
        
        self.unstructured_invoice = """
        Monthly Service Bill
        Hello Tech Solutions,
        
        This is your invoice for October 2023 services.
        The total amount due is ₹2,50,000.00
        Please contact us at support@company.com
        Invoice reference: MSI/2023/006
        """
        
        self.minimal_invoice = """
        INVOICE
        Date: 16-10-2023
        Total: ₹75,000.00
        """
        
        self.international_invoice = """
        COMMERCIAL INVOICE
        Invoice #: INT/2023/007
        Client: Global Imports Inc.
        Amount: $15,000.00
        Contact: info@globalimports.com
        """

    def test_agent_initialization(self):
        self.assertIsNotNone(self.agent)
        self.assertEqual(len(self.agent.processed_forms), 0)

    def test_amount_extraction_patterns(self):
        test_cases = [
            ("Total Amount: ₹1,53,400.00", "1,53,400.00"),
            ("Amount: $15,000.00", "15,000.00"),
            ("Total: ₹75,000.00", "75,000.00"),
        ]
        
        for text, expected_amount in test_cases:
            info = self.agent.extract_form_fields(text)
            self.assertEqual(info['amount'], expected_amount)

    def test_invoice_number_patterns(self):
        test_cases = [
            ("Invoice Number: INV/2023/001", "INV/2023/001"),
            ("Invoice #: INT/2023/007", "INT/2023/007"),
            ("Invoice reference: MSI/2023/006", "MSI/2023/006"),
        ]
        
        for text, expected_invoice in test_cases:
            info = self.agent.extract_form_fields(text)
            self.assertEqual(info['invoice_number'], expected_invoice)

    def test_question_answering(self):
        result = self.agent.answer_question(
            self.structured_invoice, 
            "What is the invoice number?"
        )
        self.assertIsNotNone(result['answer'])
        
        result = self.agent.answer_question(
            self.unstructured_invoice,
            "What is the total amount?"
        )
        self.assertIsNotNone(result['answer'])

    def test_document_summarization(self):
        summary = self.agent.summarize_document(self.structured_invoice)
        self.assertIsNotNone(summary)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 10)

    def test_empty_text_handling(self):
        summary = self.agent.summarize_document("")
        self.assertEqual(summary, "Document too short for summary")
        
        info = self.agent.extract_form_fields("")
        self.assertEqual(info['invoice_number'], None)

    def test_amount_parsing(self):
        amount = self.agent._parse_amount("1,53,400.00")
        self.assertEqual(amount, 153400.0)
        
        amount = self.agent._parse_amount("₹75,000.00")
        self.assertEqual(amount, 75000.0)

    def test_cross_form_question_answering(self):
        form1 = {
            'filename': 'test1.pdf',
            'text': self.structured_invoice,
            'info': self.agent.extract_form_fields(self.structured_invoice),
            'summary': self.agent.summarize_document(self.structured_invoice)
        }
        
        form2 = {
            'filename': 'test2.pdf', 
            'text': self.unstructured_invoice,
            'info': self.agent.extract_form_fields(self.unstructured_invoice),
            'summary': self.agent.summarize_document(self.unstructured_invoice)
        }
        
        self.agent.processed_forms = [form1, form2]
        
        result = self.agent.ask_question_about_forms("What is the invoice number?")
        self.assertIsNotNone(result['answer'])

    def test_invalid_question_handling(self):
        result = self.agent.answer_question(
            self.minimal_invoice,
            "What is the customer's favorite color?"
        )
        self.assertIsNotNone(result['answer'])

if __name__ == '__main__':
    unittest.main()
