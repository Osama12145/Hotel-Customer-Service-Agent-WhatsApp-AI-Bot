from __future__ import annotations

from fastapi import Header, HTTPException, Query, status

from app.config import get_settings


def require_admin(
    x_admin_key: str | None = Header(default=None),
    admin_key: str | None = Query(default=None),
) -> None:
    settings = get_settings()
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin API is disabled. Set ADMIN_API_KEY to enable it.",
        )
    provided_key = x_admin_key or admin_key
    if provided_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin key.",
        )


def render_admin_dashboard(
    *,
    summary: dict,
    questions: list[dict],
    gaps: list[dict],
    messages: list[dict],
) -> str:
    return f"""
<!doctype html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hotel Agent Analytics</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #17202a; }}
    header {{ background: #102a43; color: white; padding: 22px 28px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1, h2 {{ margin: 0 0 14px; }}
    section {{ margin-bottom: 26px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
    .metric, table {{ background: white; border: 1px solid #d9e2ec; border-radius: 8px; }}
    .metric {{ padding: 16px; }}
    .metric strong {{ display: block; font-size: 26px; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #edf2f7; text-align: right; vertical-align: top; }}
    th {{ background: #e9eef5; }}
    tr:last-child td {{ border-bottom: 0; }}
    .ltr {{ direction: ltr; text-align: left; }}
    .muted {{ color: #627d98; }}
  </style>
</head>
<body>
  <header>
    <h1>لوحة تحليلات مساعد الفندق</h1>
    <div class="muted">آخر {summary.get("days", 30)} يوم</div>
  </header>
  <main>
    <section class="grid">
      {metric("المحادثات", summary.get("conversations", 0))}
      {metric("الرسائل", summary.get("messages", 0))}
      {metric("Leads الحجز", summary.get("booking_leads", 0))}
      {metric("التصعيدات", summary.get("handoffs", 0))}
      {metric("فجوات المعرفة", summary.get("knowledge_gaps", 0))}
      {metric("متوسط الرد ms", summary.get("avg_response_ms", 0))}
    </section>
    <section>
      <h2>أكثر الأسئلة</h2>
      {table(["السؤال", "النية", "اللغة", "التكرار"], questions, ["user_message", "intent", "language", "count"])}
    </section>
    <section>
      <h2>فجوات المعرفة</h2>
      {table(["السؤال", "الموضوع", "التكرار", "الحالة", "اقتراح"], gaps, ["sample_question", "missing_topic", "occurrences", "status", "suggested_knowledge_section"])}
    </section>
    <section>
      <h2>آخر الرسائل</h2>
      {table(["الوقت", "رسالة العميل", "رد المساعد", "النية", "الثقة"], messages, ["created_at", "user_message", "assistant_reply", "intent", "answer_confidence"])}
    </section>
  </main>
</body>
</html>
""".strip()


def metric(label: str, value) -> str:
    return f'<div class="metric"><span>{escape(label)}</span><strong>{escape(str(value))}</strong></div>'


def table(headers: list[str], rows: list[dict], keys: list[str]) -> str:
    if not rows:
        return '<p class="muted">لا توجد بيانات حتى الآن.</p>'
    header_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body_html = ""
    for row in rows:
        cells = "".join(f"<td>{escape(str(row.get(key) or ''))}</td>" for key in keys)
        body_html += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>"


def escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
