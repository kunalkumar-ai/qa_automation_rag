import os
from openai import OpenAI
from config import OPENAI_API_KEY, CHAT_MODEL, DOCS_PATH

client = OpenAI(api_key=OPENAI_API_KEY)

DOCUMENTS = {
    "legal": [
        ("warranty_policy.txt", "ElectroTech 24-month product warranty: coverage scope, claim process, exclusions, contact details"),
        ("terms_of_service.txt", "ElectroTech terms of service: purchase conditions, liability limits, dispute resolution, governing law"),
    ],
    "quality_control": [
        ("iso_certification.txt", "ElectroTech ISO 9001:2015 certification: scope, audit schedule, quality management process, certificate validity"),
        ("defect_return_process.txt", "ElectroTech defective product return process: eligibility, RMA number, packaging instructions, refund timelines"),
    ],
    "supply_chain": [
        ("delivery_timelines.txt", "ElectroTech standard delivery timelines by region: domestic 3-5 days, EU 7-10 days, rest of world 14-21 days"),
        ("shipping_regions.txt", "ElectroTech shipping coverage: supported countries, restricted regions, courier partners, tracking information"),
    ],
    "environment_sustainability": [
        ("rohs_compliance.txt", "ElectroTech RoHS 3 compliance (EU 2015/863): restricted substances, declaration of conformity, product scope"),
        ("recycling_program.txt", "ElectroTech WEEE recycling program: drop-off points, take-back scheme, packaging recycling, annual targets"),
    ],
    "it": [
        ("firmware_support.txt", "ElectroTech firmware update policy: supported models, update frequency, how to update, end-of-life schedule"),
        ("software_compatibility.txt", "ElectroTech software compatibility: supported OS versions (Windows 10/11, macOS 13+, Ubuntu 22.04), driver downloads"),
    ],
    "energy_water": [
        ("energy_ratings.txt", "ElectroTech EU energy label ratings for product range: ET-X200 (A+), ET-X300 (A++), ET-S100 (B), test conditions"),
        ("power_consumption.txt", "ElectroTech product power consumption specs: standby wattage, active wattage, annual kWh estimates, ErP compliance"),
    ],
    "after_sales": [
        ("repair_process.txt", "ElectroTech authorised repair process: booking, turnaround times (in-warranty 5 days, out-of-warranty 10 days), courier collection"),
        ("spare_parts.txt", "ElectroTech spare parts availability: supported models, ordering process, lead times, pricing structure"),
    ],
    "product_safety": [
        ("ce_marking.txt", "ElectroTech CE marking: applicable directives (LVD, EMC, RED), notified body, declaration of conformity process"),
        ("recall_procedures.txt", "ElectroTech product recall process: how customers are notified, return instructions, replacement timelines, RAPEX reporting"),
    ],
    "finance": [
        ("payment_terms.txt", "ElectroTech B2B payment terms: net-30 standard, early payment discount 2/10, accepted methods, invoice dispute process"),
        ("invoice_policy.txt", "ElectroTech invoice policy: VAT treatment, credit note process, invoice correction requests, retention period"),
    ],
    "hr": [
        ("careers_faq.txt", "ElectroTech careers FAQ: how to apply, interview process, graduate programmes, relocation support, benefits overview"),
        ("internship_program.txt", "ElectroTech internship programme: duration (6 months), departments available, application window (January-March), stipend"),
    ],
    "rd": [
        ("product_roadmap_faq.txt", "ElectroTech product roadmap FAQ: upcoming product lines, feature request process, beta programme, NDA-covered information policy"),
        ("innovation_programs.txt", "ElectroTech innovation partnerships: university collaboration programme, startup accelerator, joint development agreements"),
    ],
    "marketing": [
        ("product_catalog.txt", "ElectroTech 2025 product catalog: ET-X200, ET-X300, ET-S100, ET-P50 — key specs, target segments, pricing tiers"),
        ("partner_guidelines.txt", "ElectroTech authorised reseller brand guidelines: logo usage, approved imagery, co-marketing rules, prohibited claims"),
    ],
    "logistics": [
        ("import_export_info.txt", "ElectroTech import/export information: HS codes, country of origin, EORI registration, export control classification"),
        ("customs_documentation.txt", "ElectroTech customs documentation: commercial invoice requirements, packing list, certificate of origin, incoterms used (DDP/DAP)"),
    ],
    "customer_support": [
        ("escalation_process.txt", "ElectroTech customer support escalation: tier 1 (email 48h SLA), tier 2 (phone, 24h SLA), tier 3 (management, 4h SLA)"),
        ("sla_response_times.txt", "ElectroTech service level agreement: response time commitments by channel and severity, measurement method, exclusions"),
    ],
}


def generate_document(topic: str) -> str:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write internal company documents for ElectroTech, an electronics manufacturer. "
                    "Write in formal business prose — no bullet points, no headers, only paragraphs. "
                    "Include realistic specifics: product model names (ET-X200, ET-X300, ET-S100, ET-P50), "
                    "regulation numbers, dates (use year 2025), policy revision numbers, and contact email addresses. "
                    "Length: approximately 400 words."
                ),
            },
            {"role": "user", "content": f"Write a company document about: {topic}"},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def main():
    for dept, docs in DOCUMENTS.items():
        dept_path = os.path.join(DOCS_PATH, dept)
        os.makedirs(dept_path, exist_ok=True)
        for filename, topic in docs:
            filepath = os.path.join(dept_path, filename)
            print(f"Generating {dept}/{filename}...")
            content = generate_document(topic)
            with open(filepath, "w") as f:
                f.write(content)
    print("All documents generated.")


if __name__ == "__main__":
    main()
