"""AutoPrice AI - Used Car Price Prediction (Flask + scikit-learn)."""
import functools
import os
import re

import joblib
import pandas as pd
from flask import (
    Flask, flash, redirect, render_template, request, session, url_for
)

import database as db

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "car_price_model.pkl")

# Must match the `current_year` used when the model was trained.
CURRENT_YEAR = 2025

VALID_FUEL = {"Petrol", "Diesel", "CNG", "LPG", "Electric"}
VALID_SELLER = {"Individual", "Dealer", "Trustmark Dealer"}
VALID_TRANSMISSION = {"Manual", "Automatic"}
VALID_OWNER = {
    "First Owner", "Second Owner", "Third Owner",
    "Fourth & Above Owner", "Test Drive Car",
}

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")

app = Flask(__name__)
app.secret_key = os.environ.get(
    "AUTOPRICE_SECRET_KEY", "autoprice-dev-secret-change-in-production"
)
app.config["DATABASE"] = os.path.join(BASE_DIR, "instance", "autoprice.db")

model = joblib.load(MODEL_PATH)

# Close the request-scoped SQLite connection after every request.
app.teardown_appcontext(db.close_db)


def init_database():
    """Create the SQLite schema + seed the default admin. Safe to call twice."""
    with app.app_context():
        db.init_db()


@app.context_processor
def inject_globals():
    """Expose shared values to every template."""
    return {
        "current_year": CURRENT_YEAR,
        "min_len": db.MIN_PASSWORD_LENGTH,
        "total_users": db.count_users(),
    }


def login_required(view):
    """Redirect anonymous visitors to the sign-in page."""
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            flash("Please sign in to access the dashboard.", "error")
            return redirect(url_for("index"))
        return view(*args, **kwargs)
    return wrapped



def _clean_feature_names(raw_names):
    """Turn pipeline feature names into short, readable dashboard labels."""
    labels = []
    for raw in raw_names:
        name = raw.split("__", 1)[-1]        # drop the 'cat__' / 'remainder__' prefix
        name = name.replace("_", " ")
        labels.append(name.strip())
    return labels


def get_importances(top_n=6):
    """Real global feature importances from the fitted Random Forest."""
    estimator = model.named_steps["model"]
    raw_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = estimator.feature_importances_

    pairs = list(zip(_clean_feature_names(raw_names), importances))
    pairs.sort(key=lambda item: item[1], reverse=True)

    total = float(sum(importances)) or 1.0
    return [(name, float(value) / total * 100.0) for name, value in pairs[:top_n]]


@app.route("/")
def index():
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Create a new account backed by the SQLite users table."""
    if session.get("logged_in"):
        return redirect(url_for("input_form"))

    form = {"full_name": "", "email": ""}

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        form = {"full_name": full_name, "email": email}

        if not full_name:
            flash("Full name is required.", "error")
        elif len(full_name) < 2:
            flash("Full name must be at least 2 characters.", "error")
        elif not EMAIL_RE.match(email):
            flash("Please enter a valid email address.", "error")
        elif not password:
            flash("Password is required.", "error")
        elif len(password) < db.MIN_PASSWORD_LENGTH:
            flash(
                f"Password must be at least {db.MIN_PASSWORD_LENGTH} characters.",
                "error",
            )
        elif password != confirm:
            flash("The two passwords do not match.", "error")
        elif db.email_exists(email):
            flash("An account with that email already exists.", "error")
        else:
            db.create_user(email, password, full_name)
            flash("Account created successfully. Please sign in.", "success")
            return redirect(url_for("index"))

    return render_template("register.html", active="register", form=form)


@app.route("/login", methods=["POST"])
def check_login():
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    user = db.authenticate(email, password)
    if user is not None:
        session["logged_in"] = True
        session["user_email"] = user["email"]
        session["user_name"] = user["full_name"]
        return redirect(url_for("input_form"))

    flash("Invalid email or password. Please try again.", "error")
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    session.pop("user_email", None)
    session.pop("user_name", None)
    return redirect(url_for("index"))


@app.route("/input")
@login_required
def input_form():
    estimator = model.named_steps["model"]
    return render_template(
        "input.html",
        active="input",
        importances=get_importances(),
        n_features=int(estimator.n_features_in_),
        n_trees=int(estimator.n_estimators),
        model_name=type(estimator).__name__,
    )



@app.route("/predict", methods=["POST"])
@login_required
def predict():
    # --- read + validate the raw form values ---
    try:
        year = int(request.form["year"])
        km_driven = int(request.form["km_driven"])
    except (KeyError, ValueError, TypeError):
        flash("Please enter a valid numeric value for year and kilometres driven.", "error")
        return redirect(url_for("input_form"))

    fuel = request.form.get("fuel", "").strip()
    seller_type = request.form.get("seller_type", "").strip()
    transmission = request.form.get("transmission", "").strip()
    owner = request.form.get("owner", "").strip()

    if not (1950 <= year <= CURRENT_YEAR):
        flash(f"Year must be between 1950 and {CURRENT_YEAR}.", "error")
        return redirect(url_for("input_form"))

    if km_driven < 0:
        flash("Kilometres driven cannot be negative.", "error")
        return redirect(url_for("input_form"))

    for label, value, allowed in (
        ("Fuel type", fuel, VALID_FUEL),
        ("Seller type", seller_type, VALID_SELLER),
        ("Transmission", transmission, VALID_TRANSMISSION),
        ("Owner", owner, VALID_OWNER),
    ):
        if value not in allowed:
            flash(f"Invalid {label.lower()} selected.", "error")
            return redirect(url_for("input_form"))

    # --- build a training-compatible feature frame ---
    input_df = pd.DataFrame([{
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "car_age": CURRENT_YEAR - year,   # engineered exactly as during training
    }])

    try:
        prediction = float(model.predict(input_df)[0])
    except Exception as exc:                      # pragma: no cover - safety net
        app.logger.exception("Prediction failed")
        flash(f"Prediction failed: {exc}", "error")
        return redirect(url_for("input_form"))

    estimator = model.named_steps["model"]
    return render_template(
        "result.html",
        active="result",
        price=prediction,
        inputs={
            "year": year,
            "km_driven": km_driven,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
        },
        importances=get_importances(),
        n_trees=int(estimator.n_estimators),
    )


@app.errorhandler(500)
def handle_server_error(error):
    """Render framework-level errors inside the dashboard theme."""
    return render_template(
        "error.html",
        code=500,
        message="An unexpected error occurred while processing your request.",
    ), 500


if __name__ == "__main__":
    init_database()   # create tables + seed admin on first run
    app.run(port=5002, debug=True)
