"""Verification suite: registration + SQLite persistence + existing ML flow."""
import os
import re
import sqlite3
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

DB_PATH = os.path.join(BASE, "instance", "autoprice.db")
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)          # start from a clean database

import app as appmod            # noqa: E402

appmod.app.config.update(TESTING=True)
appmod.init_database()

out, results = [], []


def check(label, ok, extra=""):
    out.append(f"[{'PASS' if ok else 'FAIL'}] {label} {extra}".rstrip())
    results.append(bool(ok))
    return ok


client = appmod.app.test_client()
VALID_PREDICT = {
    "year": "2018", "km_driven": "45000", "fuel": "Petrol",
    "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner",
}

# ---------------- database ----------------
check("sqlite file created", os.path.isfile(DB_PATH), DB_PATH)
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)")]
check("users table has expected columns",
      set(cols) >= {"id", "email", "password_hash", "full_name", "created_at"},
      str(cols))
seed = conn.execute("SELECT * FROM users WHERE email='admin@example.com'").fetchone()
check("default admin seeded", seed is not None)
check("admin password is hashed (not plain text)",
      seed is not None and seed["password_hash"] != "password123"
      and "pbkdf2" in seed["password_hash"].lower(),
      seed["password_hash"][:24] + "..." if seed else "no row")
idx = list(conn.execute("PRAGMA index_list(users)"))
check("email has a UNIQUE constraint (auto-index origin='u')",
      any(r["origin"] == "u" for r in idx), str([(r["name"], r["origin"]) for r in idx]))

# ---------------- register page ----------------
r = client.get("/register")
html = r.get_data(as_text=True)
check("GET /register -> 200", r.status_code == 200)
check("register page has 4 field icons", html.count("field-icon") == 4,
      f"count={html.count('field-icon')}")
for f in ["full_name", "email", "password", "confirm_password"]:
    check(f"register has input[name={f}]", f'name="{f}"' in html)
check("register has strength meter", "pw-meter" in html)
check("register has dark theme link", "style.css" in html)
check("register has car svg", "car.svg" in html)
check("register links back to sign in", 'href="/"' in html)
check("nav shows Register link", "Register" in html)

# ---------------- registration validation ----------------
base = {"full_name": "Test User", "email": "test@example.com",
        "password": "Str0ng!Pass", "confirm_password": "Str0ng!Pass"}

cases = [
    ({**base, "full_name": ""}, "Full name is required"),
    ({**base, "full_name": "A"}, "at least 2 characters"),
    ({**base, "email": "not-an-email"}, "valid email address"),
    ({**base, "email": ""}, "valid email address"),
    ({**base, "password": ""}, "Password is required"),
    ({**base, "password": "short"}, "at least 8 characters"),
    ({**base, "confirm_password": "different!"}, "do not match"),
    ({**base, "email": "admin@example.com"}, "already exists"),
]
for payload, expected in cases:
    rr = client.post("/register", data=payload, follow_redirects=True)
    body = rr.get_data(as_text=True)
    check(f"rejects: {expected}", expected in body, f"-> {expected}")

# nothing should have been persisted from failed attempts
n_now = conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]
check("failed registrations not persisted", n_now == 1, f"count={n_now}")

# ---------------- successful registration ----------------
rr = client.post("/register", data=base, follow_redirects=True)
check("valid registration succeeds", "Account created successfully" in rr.get_data(as_text=True))
row = conn.execute("SELECT * FROM users WHERE email='test@example.com'").fetchone()
check("new user row persisted", row is not None)
check("full_name stored", row is not None and row["full_name"] == "Test User")
check("password stored hashed", row is not None and row["password_hash"] != "Str0ng!Pass")
check("created_at auto-filled", row is not None and bool(row["created_at"]))
check("user count now 2", conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"] == 2)

# duplicate rejected
rr = client.post("/register", data=base, follow_redirects=True)
check("duplicate email rejected", "already exists" in rr.get_data(as_text=True))

# ---------------- login with DB accounts ----------------
r = client.post("/login", data={"email": "test@example.com", "password": "Str0ng!Pass"},
                follow_redirects=True)
html = r.get_data(as_text=True)
check("new user can sign in", "Vehicle Details" in html)
check("header shows user name", "Test User" in html)

r = client.post("/login", data={"email": "test@example.com", "password": "wrong"},
                follow_redirects=True)
check("wrong password rejected", "Invalid email or password" in r.get_data(as_text=True))

r = client.post("/login", data={"email": "ghost@example.com", "password": "whatever"},
                follow_redirects=True)
check("unknown user rejected", "Invalid email or password" in r.get_data(as_text=True))

r = client.post("/login", data={"email": "admin@example.com", "password": "password123"},
                follow_redirects=True)
check("seeded admin can still sign in", "Vehicle Details" in r.get_data(as_text=True))

# case-insensitive email
client.get("/logout")
r = client.post("/login", data={"email": "TEST@Example.com", "password": "Str0ng!Pass"},
                follow_redirects=True)
check("email match is case-insensitive", "Vehicle Details" in r.get_data(as_text=True))

# ---------------- route protection ----------------
anon = appmod.app.test_client()
r = anon.get("/input", follow_redirects=True)
check("anonymous /input is blocked", "Please sign in" in r.get_data(as_text=True))
r = anon.post("/predict", data=VALID_PREDICT, follow_redirects=True)
check("anonymous /predict is blocked", "Please sign in" in r.get_data(as_text=True))
r = anon.get("/register", follow_redirects=True)
check("anonymous can reach /register", "Create Your" in r.get_data(as_text=True))

# logged-in user cannot hit /register
r = client.get("/register", follow_redirects=True)
check("logged-in user redirected off /register", "Vehicle Details" in r.get_data(as_text=True))

# ---------------- persistence across restarts ----------------
conn.close()
appmod.init_database()
conn2 = sqlite3.connect(DB_PATH)
conn2.row_factory = sqlite3.Row
check("data survives re-init", conn2.execute(
    "SELECT COUNT(*) c FROM users").fetchone()["c"] == 2)
conn2.close()
appmod.init_database()   # idempotency check
conn3 = sqlite3.connect(DB_PATH)
conn3.row_factory = sqlite3.Row
admin_count = conn3.execute(
    "SELECT COUNT(*) c FROM users WHERE email='admin@example.com'").fetchone()["c"]
conn3.close()
check("init_database is idempotent (no dup admin)", admin_count == 1, f"count={admin_count}")

# ---------------- existing ML flow still works ----------------
PRICE_RX = r'class="price">\s*&#8377;([\d,]+)'
r = client.post("/predict", data=VALID_PREDICT)
html = r.get_data(as_text=True)
m = re.search(PRICE_RX, html)
check("predict still works after refactor", r.status_code == 200 and m is not None,
      f"price={m.group(1) if m else 'NONE'}")
check("result page still shows importance bars", html.count("fi-fill") == 6)

r = client.post("/predict", data={**VALID_PREDICT, "year": "2099"}, follow_redirects=True)
check("prediction validation still works", "must be between" in r.get_data(as_text=True))

with open("_verify.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(out) + f"\n\nSUMMARY: {sum(results)}/{len(results)} checks passed\n")
print(f"{sum(results)}/{len(results)} passed")
