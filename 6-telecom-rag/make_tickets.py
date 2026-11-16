import sqlite3

conn = sqlite3.connect("rag-app/data/tickets.db")

conn.execute("""
    CREATE TABLE IF NOT EXISTS tickets (
        ticket_id   INTEGER PRIMARY KEY,
        issue_type  TEXT,
        description TEXT,
        resolution  TEXT,
        category    TEXT,
        status      TEXT
    )
""")

rows = [
    ("SIM not detected",
     "Phone shows SIM not detected after a restart.",
     "Remove and reinsert the SIM, then restart the phone. If it persists, replace the SIM at a service centre.",
     "Device", "resolved"),
    ("Roaming charge",
     "Customer was charged for roaming while a bundle was active.",
     "Verified the bundle covered the destination and refunded the roaming charge.",
     "Billing", "resolved"),
    ("Wi-Fi calling",
     "Customer cannot enable Wi-Fi calling.",
     "Enabled the feature on the account and asked the customer to turn it on in phone settings.",
     "Network", "resolved"),
    ("Slow data",
     "Mobile data is very slow in the evening.",
     "Advised a network reset and checked for congestion in the area.",
     "Network", "open"),
]

conn.executemany(
    "INSERT INTO tickets (issue_type, description, resolution, category, status) VALUES (?, ?, ?, ?, ?)",
    rows,
)
conn.commit()
conn.close()
print("tickets table created with", len(rows), "rows")