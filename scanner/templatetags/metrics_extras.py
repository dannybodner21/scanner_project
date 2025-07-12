# scanner/templatetags/metrics_extras.py

from django import template

register = template.Library()

@register.filter
def percent_change(current, entry):
    try:
        current = float(current)
        entry = float(entry)
        return ((current - entry) / entry) * 100
    except (ValueError, ZeroDivisionError, TypeError):
        return None
