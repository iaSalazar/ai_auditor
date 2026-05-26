# Dataset folder

The dataset for the audit assistant lives at `data/northstar_robotics_audit_dataset/` (this folder's subdirectory). It's **not included in this repository** for two reasons:

1. **Copyright** — the IAASB handbook PDF is copyrighted by IFAC; redistribution requires permission. Free to download for personal use from https://www.ifac.org.
2. **Proprietary test material** — the Northstar Robotics audit data is fictional but proprietary; not shipped here to avoid leaking it.

## Required directory layout

Create the subdirectory `data/northstar_robotics_audit_dataset/` and place the **12 files** below inside it. The path is bind-mounted read-only into the `audit-app` container at `/app/data` — once the files are in place, `make up` will pick them up automatically (ingestion runs idempotently at startup, ~70 seconds first time).

```
data/
├── README.md                                  ← you are here
└── northstar_robotics_audit_dataset/          ← create this folder, drop files in
    │
    │   ─── STRUCTURED (5 files → SQLite, 425 rows total) ───
    ├── financial_transactions.csv             # 177 rows — ledger transactions
    ├── journal_entries.csv                    # 44 rows — debit/credit entries
    ├── audit_support_mapping.csv              # 177 rows — TX ↔ supporting document
    ├── vendor_master.json                     # 10 vendors with metadata
    ├── trial_balance.xlsx                     # account-level balances
    │
    │   ─── UNSTRUCTURED (7 files → ChromaDB, 722 chunks tagged with doc_type) ───
    ├── revenue_recognition_policy.pdf         # doc_type: policy
    ├── audit_planning_memo.docx               # doc_type: memo
    ├── audit_procedures_revenue_and_expenses.docx   # doc_type: procedure
    ├── client_provided_evidence_notes.txt     # doc_type: evidence
    ├── travel_expense_workpaper.md            # doc_type: workpaper
    ├── Retail-Lease-Agreement-Acme.pdf        # doc_type: contract
    └── IAASB-2023-2024-Handbook-Volume-2.pdf  # doc_type: standard — 134 pages, ~2.6 MB
```

## File summary

| Side | Files | Loaded into | Used by |
|---|---|---|---|
| **Structured** | 5 (.csv / .json / .xlsx) | SQLite, 5 tables auto-named after the source files | SQL retriever — handles transaction lookups, totals, vendor queries, journal entries, balances |
| **Unstructured** | 7 (PDF / DOCX / TXT / MD) | ChromaDB, 722 chunks (1000 chars / 200 overlap), each tagged with a `doc_type` metadata field | Vector retriever — handles policy/standards/evidence/procedure questions; `doc_type` filter enables the richer routing dictionary's structural retrieval scoping |

## Verifying the setup

After placing all 12 files and running `make up`, the startup logs should show:

```
Ingesting structured audit data into SQLite…
  transactions: 177 rows
  journal_entries: 44 rows
  support_mapping: 177 rows
  vendors: 10 rows
  trial_balance: 17 rows
Structured ingestion complete.

Ingesting unstructured documents into ChromaDB…
  loaded 134 pages from IAASB-2023-2024-Handbook-Volume-2.pdf [standard]
  loaded 1 pages from revenue_recognition_policy.pdf [policy]
  loaded 28 pages from Retail-Lease-Agreement-Acme.pdf [contract]
  loaded 1 pages from audit_planning_memo.docx [memo]
  loaded 1 pages from audit_procedures_revenue_and_expenses.docx [procedure]
  loaded 1 pages from client_provided_evidence_notes.txt [evidence]
  loaded 1 pages from travel_expense_workpaper.md [workpaper]
Stored 722 chunks in ChromaDB collection 'audit_documents'.
```

If a file is missing you'll see `missing file: <filename>` and that file's content won't be retrievable. The system still runs but the eval cases that depend on the missing file will fail.

## Quick smoke-test once the dataset is in place

```bash
make ask Q="What is the total travel expense for Q1 2026?"
# expected: returns ~$112,590.75 with [account 6100] citation
```
