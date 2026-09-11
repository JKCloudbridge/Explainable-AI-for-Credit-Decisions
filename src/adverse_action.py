"""ECOA / Regulation B - style adverse-action notice generator.

**This is a capstone-project demonstration, not a real credit decision or the
notice of any real financial institution.** Regulation B (12 CFR 1002.9), which
implements the Equal Credit Opportunity Act, requires a creditor that denies an
application to give the applicant a statement of the *specific, principal
reasons* for the denial (not just "your score was too low"). This module turns
the Phase 2 `reason_codes()` output into that statement.

Run a worked example::

    python -m src.adverse_action
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.config import DECISION_THRESHOLD
from src.explain import reason_codes

NOTICE_HEADER = "NOTICE OF ACTION TAKEN AND STATEMENT OF REASONS (DEMONSTRATION)"
REG_B_CITE = "Modeled on the Equal Credit Opportunity Act / Regulation B, 12 CFR Sec. 1002.9"

BOILERPLATE_APPROVED = (
    "Your application has been approved, subject to the credit terms offered separately."
)

# Paraphrased, for educational purposes, from the applicant-rights language
# Regulation B requires in an adverse-action notice.
BOILERPLATE_DENIED_RIGHTS = (
    "You have the right to a statement of the specific reasons why your application "
    "was denied; this notice provides that statement. Within 60 days you may request "
    "additional information about this decision. If you believe you were "
    "discriminated against on a prohibited basis, you may contact the relevant "
    "regulator."
)


def generate_notice(applicant, applicant_ref: str = "N/A", k: int = 4, threshold: float | None = None) -> dict:
    """Build a structured adverse-action notice for one applicant.

    Returns the decision, probability, structured reasons, and a ready-to-print
    ``body`` string.
    """
    threshold = DECISION_THRESHOLD if threshold is None else threshold
    rc = reason_codes(applicant, k=k, threshold=threshold)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    preamble = (
        f"{NOTICE_HEADER}\n{REG_B_CITE}\n\n"
        f"Application reference: {applicant_ref}\nDate: {generated_at}\n\n"
    )

    if rc["decision"] == "APPROVE":
        body = (
            f"{preamble}Decision: APPROVED  "
            f"(model P(default) = {rc['probability']:.1%}, threshold = {threshold:.0%})\n\n"
            f"{BOILERPLATE_APPROVED}\n"
        )
    else:
        if rc["reasons"]:
            reasons_text = "\n".join(
                f"  {i + 1}. {r['statement']}" for i, r in enumerate(rc["reasons"])
            )
        else:
            reasons_text = "  (no single feature dominated; a combination of factors contributed)"
        body = (
            f"{preamble}Decision: DENIED  "
            f"(model P(default) = {rc['probability']:.1%}, threshold = {threshold:.0%})\n\n"
            f"Principal reason(s) for this decision:\n{reasons_text}\n\n"
            f"{BOILERPLATE_DENIED_RIGHTS}\n"
        )

    return {
        "decision": rc["decision"],
        "probability": rc["probability"],
        "threshold": threshold,
        "reasons": rc["reasons"],
        "generated_at": generated_at,
        "applicant_ref": applicant_ref,
        "body": body,
    }


if __name__ == "__main__":
    from src.explain import _pick_examples

    approve, deny = _pick_examples()
    for ref, applicant in (("demo-approve-001", approve), ("demo-deny-001", deny)):
        notice = generate_notice(applicant, applicant_ref=ref)
        print(notice["body"])
        print("-" * 70)
