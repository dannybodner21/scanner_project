from scanner.models import RickisMetrics
from django.utils.timezone import make_aware
from datetime import datetime

start = make_aware(datetime(2025, 3, 22))
end = make_aware(datetime(2025, 4, 12))

qs = RickisMetrics.objects.filter(
    timestamp__gte=start,
    timestamp__lt=end
)

total = qs.count()
print(f"🔄 Resetting {total} RickisMetrics entries...")

batch = []
i = 0

for m in qs:
    m.long_result = None
    m.short_result = None
    batch.append(m)
    i += 1

    if len(batch) >= 100:
        RickisMetrics.objects.bulk_update(batch, ["long_result", "short_result"])
        batch = []

    if i % 10000 == 0:
        print(f"✅ Processed {i} of {total}")

# Final batch
if batch:
    RickisMetrics.objects.bulk_update(batch, ["long_result", "short_result"])

print("✅ All long_result and short_result fields reset to None.")
