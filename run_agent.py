import sys
import os

sys.path.append('src')
from main import IntelligentFormAgent

class SmartInvoiceReader:
    def __init__(self):
        self.agent = IntelligentFormAgent()
        
        self.question_mappings = {
            'invoice_number': ['invoice no', 'invoice number', 'inv no', 'document number', 'bill number', 'inv/', 'invoice#', 'what is the invoice', 'invoice'],
            'total_amount': ['amount', 'total', 'invoice value', 'grand total', 'final amount', 'how much', 'what is the total', 'bill amount', 'rupees', '₹', 'total due', 'complete amount', 'final total'],
            'customer': ['customer', 'client', 'buyer', 'who is the customer', 'name of customer', 'invoice to', 'purchaser', 'bill to', 'client name', 'who is this invoice to'],
            'date': ['date', 'when', 'invoice date', 'date of invoice', 'issued on', 'billing date', 'what date', 'invoice issued'],
            'seller': ['seller', 'vendor', 'company', 'from', 'sent by', 'issued by', 'supplier', 'who issued', 'billing company'],
            'services': ['services', 'items', 'products', 'description', 'what is included', 'list of', 'work', 'scope', 'what services', 'items included'],
            'gst': ['gst', 'gstin', 'tax number', 'gst number', 'gstin number', 'tax id', 'gst identification'],
            'due_date': ['due date', 'due on', 'payment due', 'deadline', 'when is payment due', 'last date'],
            'bank': ['bank', 'account', 'payment information', 'ifsc', 'bank details', 'bank account', 'payment details'],
            'summary': ['summary', 'overview', 'details', 'information', 'key points', 'main points', 'tell me about', 'describe this invoice', 'what is this invoice about']
        }
    
    def find_question_match(self, user_question):
        user_question = user_question.lower().strip()
        
        clean_question = user_question.replace('?', '').strip()
        
        for standard_question, variations in self.question_mappings.items():
            for variation in variations:
                if variation in user_question:
                    return standard_question
                    
        question_words = set(clean_question.split())
        for standard_question, variations in self.question_mappings.items():
            for variation in variations:
                variation_words = set(variation.split())
                if variation_words.intersection(question_words):
                    return standard_question
                    
        return None
    
    def process_invoice(self, pdf_path):
        result = self.agent.process_document(pdf_path)
        return result
    
    def ask_smart_question(self, question):
        direct_answer = self.agent.ask_question_about_forms(question)
        
        if direct_answer['confidence'] < 0.3 and any(word in question.lower() for word in ['summary', 'overview', 'describe', 'tell me about']):
            return self._generate_structured_summary()
        
        if direct_answer['confidence'] > 0.3:
            return direct_answer
            
        best_match = self.find_question_match(question)
        
        if best_match:
            mapped_answer = self.agent.ask_question_about_forms(best_match)
            if mapped_answer['confidence'] > direct_answer['confidence']:
                return mapped_answer
        
        return direct_answer
    
    def _generate_structured_summary(self):
        """Generate a structured summary when direct Q&A fails"""
        result = self.get_last_processed_result()
        
        if not result or 'info' not in result:
            return {'answer': 'No invoice data available to generate summary', 'confidence': 0.1}
        
        info = result['info']
        summary_parts = []
        
        if info.get('invoice_number'):
            summary_parts.append(f"Invoice: {info['invoice_number']}")
        if info.get('date'):
            summary_parts.append(f"Date: {info['date']}")
        if info.get('name'):
            summary_parts.append(f"Customer: {info['name']}")
        if info.get('amount'):
            summary_parts.append(f"Total Amount: {info['amount']}")
        if info.get('seller'):
            summary_parts.append(f"Seller: {info['seller']}")
        if info.get('gst'):
            summary_parts.append(f"GST: {info['gst']}")
        
        if summary_parts:
            summary = "Invoice Summary:\n" + "\n".join(summary_parts)
            return {'answer': summary, 'confidence': 0.8}
        else:
            return {'answer': 'Insufficient data to generate summary', 'confidence': 0.2}
    
    def get_last_processed_result(self):
        """Get the last processed document result"""
        if hasattr(self.agent, 'last_result'):
            return self.agent.last_result
        return None

    def generate_invoice_summary(self, result):
        info = result['info']
        summary_parts = []
        
        if info.get('invoice_number') and len(info['invoice_number']) > 3:
            summary_parts.append(f"Invoice Number: {info['invoice_number']}")
        
        if info.get('date') and len(info['date']) > 5:
            summary_parts.append(f"Date: {info['date']}")
        
        if info.get('amount') and len(info['amount']) > 1:
            summary_parts.append(f"Amount: {info['amount']}")
        
        invalid_names = ['address', 'no', 'none', 'name', 'customer']
        if info.get('name') and info['name'].lower() not in invalid_names:
            summary_parts.append(f"Customer: {info['name']}")
        
        if info.get('email') and '@' in info['email'] and '.' in info['email']:
            summary_parts.append(f"Email: {info['email']}")
        
        if info.get('phone') and len(info['phone']) > 7 and len(info['phone']) < 20:
            summary_parts.append(f"Phone: {info['phone']}")
        
        if summary_parts:
            return "\n".join(summary_parts)
        else:
            return f"Document Summary: {result.get('summary', 'No summary available')}"

def main():
    print("\nSmart Invoice Reader")
    print("====================\n")
    
    pdf_name = input("Enter PDF file name from data folder: ").strip()
    pdf_path = f'data/{pdf_name}'
    
    if not os.path.exists(pdf_path):
        print("Error: PDF file not found in data folder")
        return
    
    reader = SmartInvoiceReader()
    result = reader.process_invoice(pdf_path)
    
    if not result:
        print("Error: Could not process the PDF file")
        return
    
    print("PDF processed successfully")
    
    print("\nInvoice Summary")
    print("---------------")
    summary = reader.generate_invoice_summary(result)
    print(summary)
    
    print("\nQuestion Answering Mode")
    print("-----------------------")
    print("You can ask questions like:")
    print("- What is the total amount?")
    print("- Who is the customer?")
    print("- What is the invoice date?")
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("Enter your question: ").strip()
        
        if question.lower() in ['quit', 'exit']:
            print("Thank you for using Smart Invoice Reader")
            break
            
        if not question:
            continue
        
        answer_data = reader.ask_smart_question(question)
        answer = answer_data['answer']
        confidence = answer_data['confidence']
        
        print(f"Answer: {answer}")
        
        if confidence < 0.2:
            print("Note: Low confidence in this answer")
            print("Tip: Try rephrasing your question")

if __name__ == "__main__":
    main()
