"""
Generate multiple test financial PDFs for the Unstructured ParseQuery pipeline
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from datetime import datetime
import os


def generate_credit_card_statement(output_path: str):
    """Generate a credit card statement PDF"""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#003d7a'))

    story.append(Paragraph("CREDIT CARD STATEMENT", title_style))
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("<b>Account Holder:</b> Jennifer L. Thompson", styles['Normal']))
    story.append(Paragraph("<b>Card Number:</b> **** **** **** 4589", styles['Normal']))
    story.append(Paragraph("<b>Statement Period:</b> October 1 - October 31, 2024", styles['Normal']))
    story.append(Paragraph("<b>Statement Date:</b> November 5, 2024", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("<b>Account Summary</b>", styles['Heading2']))
    summary_data = [
        ['Previous Balance', '$2,458.92'],
        ['Payments/Credits', '-$2,458.92'],
        ['Purchases', '$3,842.15'],
        ['Cash Advances', '$500.00'],
        ['Fees Charged', '$0.00'],
        ['Interest Charged', '$0.00'],
        ['New Balance', '$3,842.15'],
        ['Minimum Payment Due', '$125.00'],
        ['Payment Due Date', 'November 28, 2024']
    ]

    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e6f2ff')),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("<b>Recent Transactions</b>", styles['Heading2']))
    trans_data = [
        ['Date', 'Description', 'Amount'],
        ['10/02', 'Amazon.com Purchase', '$127.43'],
        ['10/05', 'Whole Foods Market', '$85.29'],
        ['10/08', 'Shell Gas Station', '$52.10'],
        ['10/12', 'United Airlines', '$485.00'],
        ['10/15', 'Marriott Hotels', '$342.50'],
    ]

    trans_table = Table(trans_data, colWidths=[1*inch, 3*inch, 1.5*inch])
    trans_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003d7a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(trans_table)

    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>Contact Information</b>", styles['Normal']))
    story.append(Paragraph("Customer Service: 1-800-555-0199", styles['Normal']))
    story.append(Paragraph("Email: customerservice@premiumcard.com", styles['Normal']))
    story.append(Paragraph("SSN on File: ***-**-5821", styles['Normal']))

    doc.build(story)
    print(f"Generated: {output_path}")


def generate_bank_statement(output_path: str):
    """Generate a bank statement PDF"""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#0066cc'))

    story.append(Paragraph("MONTHLY BANK STATEMENT", title_style))
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("<b>Account Information</b>", styles['Heading2']))
    story.append(Paragraph("<b>Account Holder:</b> Michael R. Davidson", styles['Normal']))
    story.append(Paragraph("<b>Account Type:</b> Business Checking", styles['Normal']))
    story.append(Paragraph("<b>Account Number:</b> ****-****-7823", styles['Normal']))
    story.append(Paragraph("<b>Routing Number:</b> 121000248", styles['Normal']))
    story.append(Paragraph("<b>Statement Period:</b> October 1 - October 31, 2024", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("<b>Account Summary</b>", styles['Heading2']))
    summary_data = [
        ['Beginning Balance (10/01)', '$48,521.45'],
        ['Total Deposits', '$125,842.30'],
        ['Total Withdrawals', '$118,234.12'],
        ['Service Charges', '$25.00'],
        ['Ending Balance (10/31)', '$56,104.63'],
    ]

    summary_table = Table(summary_data, colWidths=[3.5*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e6f2ff')),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("<b>Contact Information</b>", styles['Normal']))
    story.append(Paragraph("First National Bank", styles['Normal']))
    story.append(Paragraph("1234 Main Street, New York, NY 10001", styles['Normal']))
    story.append(Paragraph("Phone: (212) 555-0150", styles['Normal']))
    story.append(Paragraph("Email: michael.davidson@email.com", styles['Normal']))
    story.append(Paragraph("Tax ID: 45-8921456", styles['Normal']))

    doc.build(story)
    print(f"Generated: {output_path}")


def generate_investment_portfolio(output_path: str):
    """Generate an investment portfolio statement PDF"""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#2e7d32'))

    story.append(Paragraph("INVESTMENT PORTFOLIO STATEMENT", title_style))
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("<b>Investor Information</b>", styles['Heading2']))
    story.append(Paragraph("<b>Name:</b> Patricia A. Williams", styles['Normal']))
    story.append(Paragraph("<b>Account Number:</b> INV-2024-895642", styles['Normal']))
    story.append(Paragraph("<b>SSN:</b> ***-**-3947", styles['Normal']))
    story.append(Paragraph("<b>Statement Date:</b> November 15, 2024", styles['Normal']))
    story.append(Paragraph("<b>Address:</b> 789 Oak Avenue, San Francisco, CA 94102", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("<b>Portfolio Summary</b>", styles['Heading2']))
    portfolio_data = [
        ['Asset Class', 'Value', 'Allocation'],
        ['Stocks', '$284,521.00', '55%'],
        ['Bonds', '$129,845.00', '25%'],
        ['Mutual Funds', '$77,234.00', '15%'],
        ['Cash/Money Market', '$25,900.00', '5%'],
        ['Total Portfolio Value', '$517,500.00', '100%'],
    ]

    portfolio_table = Table(portfolio_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
    portfolio_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e7d32')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e8f5e9')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ]))
    story.append(portfolio_table)
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("<b>Year-to-Date Performance</b>", styles['Normal']))
    story.append(Paragraph("Total Return: +12.4%", styles['Normal']))
    story.append(Paragraph("Total Gains: $64,320.00", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("<b>Financial Advisor</b>", styles['Normal']))
    story.append(Paragraph("Wealth Management Partners Inc.", styles['Normal']))
    story.append(Paragraph("Advisor: James K. Roberts, CFP", styles['Normal']))
    story.append(Paragraph("Phone: (415) 555-0188", styles['Normal']))
    story.append(Paragraph("Email: j.roberts@wealthpartners.com", styles['Normal']))

    doc.build(story)
    print(f"Generated: {output_path}")


def generate_mortgage_statement(output_path: str):
    """Generate a mortgage statement PDF"""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#d32f2f'))

    story.append(Paragraph("RESIDENTIAL MORTGAGE STATEMENT", title_style))
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("<b>Borrower Information</b>", styles['Heading2']))
    story.append(Paragraph("<b>Primary Borrower:</b> David J. Anderson", styles['Normal']))
    story.append(Paragraph("<b>Co-Borrower:</b> Lisa M. Anderson", styles['Normal']))
    story.append(Paragraph("<b>Property Address:</b> 456 Elm Street, Austin, TX 78701", styles['Normal']))
    story.append(Paragraph("<b>Loan Number:</b> MTG-2019-458921", styles['Normal']))
    story.append(Paragraph("<b>SSN (Primary):</b> ***-**-6742", styles['Normal']))
    story.append(Paragraph("<b>Statement Date:</b> November 1, 2024", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("<b>Loan Summary</b>", styles['Heading2']))
    loan_data = [
        ['Original Loan Amount', '$425,000.00'],
        ['Interest Rate', '4.25% Fixed'],
        ['Loan Term', '30 Years'],
        ['Loan Origination Date', 'March 15, 2019'],
        ['Current Principal Balance', '$387,542.18'],
        ['Monthly Payment (P&I)', '$2,091.15'],
        ['Escrow Payment', '$658.33'],
        ['Total Monthly Payment', '$2,749.48'],
    ]

    loan_table = Table(loan_data, colWidths=[3*inch, 2.5*inch])
    loan_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ]))
    story.append(loan_table)
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("<b>Contact Information</b>", styles['Normal']))
    story.append(Paragraph("Homestead Mortgage Corporation", styles['Normal']))
    story.append(Paragraph("Customer Service: 1-888-555-0177", styles['Normal']))
    story.append(Paragraph("Email: david.anderson@email.com", styles['Normal']))
    story.append(Paragraph("Phone: (512) 555-0194", styles['Normal']))

    doc.build(story)
    print(f"Generated: {output_path}")


def generate_insurance_policy(output_path: str):
    """Generate an insurance policy PDF"""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#1565c0'))

    story.append(Paragraph("LIFE INSURANCE POLICY DECLARATION", title_style))
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("<b>Policy Information</b>", styles['Heading2']))
    story.append(Paragraph("<b>Policy Number:</b> LIFE-2022-789456", styles['Normal']))
    story.append(Paragraph("<b>Policy Effective Date:</b> January 1, 2022", styles['Normal']))
    story.append(Paragraph("<b>Policy Anniversary Date:</b> January 1, 2025", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("<b>Insured Person</b>", styles['Heading2']))
    story.append(Paragraph("<b>Name:</b> Elizabeth R. Martinez", styles['Normal']))
    story.append(Paragraph("<b>Date of Birth:</b> April 15, 1982", styles['Normal']))
    story.append(Paragraph("<b>SSN:</b> ***-**-8934", styles['Normal']))
    story.append(Paragraph("<b>Address:</b> 321 Pine Road, Denver, CO 80202", styles['Normal']))
    story.append(Paragraph("<b>Email:</b> elizabeth.martinez@email.com", styles['Normal']))
    story.append(Paragraph("<b>Phone:</b> (303) 555-0165", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("<b>Coverage Details</b>", styles['Heading2']))
    coverage_data = [
        ['Coverage Type', 'Benefit Amount'],
        ['Death Benefit', '$750,000.00'],
        ['Accidental Death Benefit', '$1,500,000.00'],
        ['Annual Premium', '$2,845.00'],
    ]

    coverage_table = Table(coverage_data, colWidths=[3*inch, 2.5*inch])
    coverage_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1565c0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(coverage_table)
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("<b>Beneficiaries</b>", styles['Heading2']))
    story.append(Paragraph("<b>Primary:</b> Carlos J. Martinez (Spouse) - 100%", styles['Normal']))
    story.append(Paragraph("<b>Contingent:</b> Emma Martinez (Daughter) - 50%, Lucas Martinez (Son) - 50%", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("<b>Insurance Provider</b>", styles['Normal']))
    story.append(Paragraph("Guardian Life Insurance Company", styles['Normal']))
    story.append(Paragraph("Policy Services: 1-877-555-0133", styles['Normal']))

    doc.build(story)
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    # Create pdfs directory if it doesn't exist
    os.makedirs("pdfs", exist_ok=True)

    # Generate unique suffix using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate all test PDFs
    print("\nGenerating Financial Test PDFs...")
    print("=" * 50)

    generate_credit_card_statement(f"pdfs/credit_card_statement_{timestamp}.pdf")
    generate_bank_statement(f"pdfs/bank_statement_{timestamp}.pdf")
    generate_investment_portfolio(f"pdfs/investment_portfolio_{timestamp}.pdf")
    generate_mortgage_statement(f"pdfs/mortgage_statement_{timestamp}.pdf")
    generate_insurance_policy(f"pdfs/insurance_policy_{timestamp}.pdf")

    print("=" * 50)
    print(f"\nSuccessfully generated 5 financial test PDFs in pdfs/ folder")
    print(f"\nGenerated files (timestamp: {timestamp}):")
    print(f"  1. credit_card_statement_{timestamp}.pdf - Credit card with transactions and account details")
    print(f"  2. bank_statement_{timestamp}.pdf - Business checking account statement")
    print(f"  3. investment_portfolio_{timestamp}.pdf - Investment portfolio with holdings")
    print(f"  4. mortgage_statement_{timestamp}.pdf - Residential mortgage loan statement")
    print(f"  5. insurance_policy_{timestamp}.pdf - Life insurance policy declaration")
    print("\nAll PDFs contain realistic PII for testing de-identification!")
