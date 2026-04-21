"""
tools/datetime_tool.py — Date and time tool.

The model generates: <tool>datetime(query)</tool>
Supports queries like "today", "year", "days_until(2025-12-25)", etc.
"""

from datetime import datetime, date
import re


def run(query: str) -> str:
    """Tool entry point for date/time queries."""
    query = query.strip().lower()
    now = datetime.now()

    if query in ("today", "date", "current date"):
        return f"Today's date is {now.strftime('%A, %B %d, %Y')}."

    elif query in ("time", "current time", "now"):
        return f"Current time is {now.strftime('%H:%M:%S')}."

    elif query in ("year", "current year"):
        return f"Current year is {now.year}."

    elif query in ("month", "current month"):
        return f"Current month is {now.strftime('%B')} ({now.month})."

    elif query.startswith("days_until("):
        # Parse: days_until(YYYY-MM-DD)
        match = re.search(r"days_until\((\d{4}-\d{2}-\d{2})\)", query)
        if match:
            target = date.fromisoformat(match.group(1))
            delta = (target - now.date()).days
            if delta > 0:
                return f"There are {delta} days until {target}."
            elif delta == 0:
                return f"{target} is today!"
            else:
                return f"{target} was {-delta} days ago."
        return "Invalid days_until format. Use: days_until(YYYY-MM-DD)"

    elif query.startswith("day_of_week("):
        match = re.search(r"day_of_week\((\d{4}-\d{2}-\d{2})\)", query)
        if match:
            d = date.fromisoformat(match.group(1))
            return f"{d} is a {d.strftime('%A')}."
        return "Invalid format. Use: day_of_week(YYYY-MM-DD)"

    else:
        return f"Date/time info: {now.strftime('%A, %B %d, %Y, %H:%M:%S')}."
