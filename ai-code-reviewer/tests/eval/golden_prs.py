"""
golden_prs.py -- the ground truth for our test PR.

These are the issues we deliberately introduced into the
feature/user-updates branch of sample-python-api. The eval
harness checks that the agent found each one.

Each entry defines:
    category:  which agent should catch it (security/logic/design)
    keywords:  words that must appear in the issue description
               or suggested fix -- at least one must match
    severity:  minimum severity level expected
    description: human-readable explanation of what we planted

Why keywords instead of exact matching?
    The agent phrases things differently each run. "SQL injection"
    might appear as "sql injection", "SQL-injection", or "SQLi".
    Keyword matching is robust to phrasing variation while still
    checking for the substance of the finding.
"""

GOLDEN_ISSUES = [
    # --- Security issues we planted ---
    {
        "id": "hardcoded_api_key",
        "category": "security",
        "severity": "high",
        "keywords": ["INTERNAL_API_KEY", "hardcoded", "hard-coded", "hard‑coded", "environment"],
        "description": "Hardcoded INTERNAL_API_KEY constant in auth_service.py",
    },
    {
        "id": "sql_injection_authenticate",
        "category": "security",
        "severity": "critical",
        "keywords": ["sql injection", "sql-injection", "f-string", "f‑string", "parameteris", "parameteriz", "authenticate_user"],
        "description": "SQL injection in authenticate_user via f-string query",
    },
    {
        "id": "sql_injection_get_user_data",
        "category": "security",
        "severity": "medium",
        "keywords": ["get_user_data", "sql injection", "sql-injection", "user_id", "parameteris", "parameteriz"],
        "description": "SQL injection in get_user_data via f-string query",
    },
    {
        "id": "sql_injection_delete_user",
        "category": "security",
        "severity": "medium",
        "keywords": ["delete_user", "sql injection", "sql-injection", "parameteris", "parameteriz"],
        "description": "SQL injection in delete_user via f-string query",
    },
    {
        "id": "md5_password_hashing",
        "category": "security",
        "severity": "high",
        "keywords": ["md5", "MD5", "hash_password", "bcrypt", "argon", "passlib"],
        "description": "MD5 used for password hashing in user_service.py",
    },
    {
        "id": "missing_auth_delete_endpoint",
        "category": "security",
        "severity": "high",
        "keywords": ["remove_user", "delete", "authentication", "authorization", "api_key", "validate_internal"],
        "description": "No authentication on DELETE /users/{user_id} endpoint",
    },
    {
        "id": "missing_auth_admin_endpoint",
        "category": "security",
        "severity": "high",
        "keywords": ["admin", "raw", "authentication", "authorization", "hashed_password", "sensitive"],
        "description": "No authentication on GET /admin/users/{user_id}/raw endpoint",
    },

    # --- Logic issues we planted ---
    {
        "id": "delete_endpoint_not_deleting",
        "category": "logic",
        "severity": "high",
        "keywords": ["remove_user", "delete_user", "never", "not", "actually", "fetch"],
        "description": "DELETE endpoint fetches user but never calls delete_user",
    },
    {
        "id": "none_crash_get_user_data",
        "category": "logic",
        "severity": "high",
        "keywords": ["None", "null", "TypeError", "fetchone", "get_user_data", "missing"],
        "description": "get_user_data crashes with TypeError when user does not exist",
    },
    {
        "id": "plaintext_password_comparison",
        "category": "logic",
        "severity": "high",
        "keywords": ["plaintext", "plain text", "plain-text", "hash", "authenticate_user", "always fail"],
        "description": "authenticate_user compares plain password to hashed value -- always fails",
    },
]


# --- The PR we test against ---
TEST_PR = {
    "repo": "praveenkumar993/sample-python-api",
    "pr_number": 1,
}