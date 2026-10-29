# تشخیص علت کندی آموزش reference

اندازه‌گیری GPU utilization یا ETA به‌تنهایی علت کندی را مشخص نمی‌کند. این ابزار چند گام واقعی آموزش با همان checkpoint، batch، داده، امبدینگ، loss و reference backend را در یک process جدا اجرا می‌کند. وزن‌ها فقط در حافظه همان process تغییر می‌کنند؛ checkpointهای آموزش اصلی بازنویسی نمی‌شوند.

## آنچه از کد مشخص است

در `AllAtomEncoder` توالی residueها به atom14 باز می‌شود: طول SRU برابر `(N+M)*14` است. با دو patch پنجاه‌تایی، طول ۱۴۰۰ می‌شود. frame averaging هشت frame دارد و batch مؤثر در encoder برابر `B*8` است؛ در batch=16، این عدد ۱۲۸ است. SRU این checkpoint یک لایه و دوطرفه است. reference recurrence در طول توالی حلقه می‌زند؛ اجرای Python به معنی CPU بودن tensorها نیست، اما می‌تواند تعداد زیادی عملیات کوچک CUDA ایجاد کند.

در loss نیز ابتدا `autograd.grad(energy, coordinates, create_graph=True)` و سپس `loss.backward()` اجرا می‌شود. بنابراین هزینه با forward/backward معمولی یک مدل affinity یکسان نیست. اینکه کدام بخش واقعاً غالب است، باید از خروجی ابزار مشخص شود؛ این توضیح جای اندازه‌گیری را نمی‌گیرد.

## اجرا روی سرور

۱. در ترمینال دوم، وجود آخرین checkpoint را بررسی کنید:

```bash
cd /workspace/mohsen/DSM-bind-TCR-pMHC
ls -lh outputs/reference_full_b16_01/last.pt
```

۲. وقتی checkpoint موجود است، آموزش را در ترمینال خودش با Ctrl+C متوقف کنید. profiler و آموزش نباید همزمان روی همان GPU اجرا شوند؛ سرعت اندازه‌گیری‌شده مخدوش می‌شود.

۳. نسخه جدید را بگیرید و profiler را اجرا کنید:

```bash
git pull --ff-only origin main
python scripts/profile_reference_training.py \
  --run-dir outputs/reference_full_b16_01 \
  --output-dir outputs/profile_reference_b16_01
```

یک batch برای warmup، دو batch برای زمان‌گیری بدون tracing و یک batch دیگر برای profiler اجرا می‌شود. tracing می‌تواند سربار و مصرف RAM قابل‌توجهی داشته باشد؛ زمان batch پروفایل‌شده در تخمین سرعت استفاده نمی‌شود. امبدینگ جدید محاسبه یا دانلود نمی‌شود. جدول operatorها همیشه ذخیره می‌شود؛ `--trace` اختیاری فایل Chrome trace بزرگ‌تری تولید می‌کند.

## خروجی لازم برای بررسی

```text
outputs/profile_reference_b16_01/
  timings.json
  operators_cpu.txt
  operators_cuda.txt
```

در `timings.json` این مراحل جدا زمان‌گیری می‌شوند:

- خواندن pair و مختصات از SQLite؛
- lookup و decompress امبدینگ، ساخت batch و انتقال به GPU؛
- forward مدل شامل محاسبه مشتق انرژی نسبت به مختصات؛
- backward loss و مشتق‌های مختلط؛
- بررسی و clipping گرادیان؛
- optimizer.step در حافظه process تشخیصی.

مرز مراحل با CUDA synchronize جدا می‌شود؛ این زمان‌ها برای تشخیص اجزای گام هستند. validation و نوشتن checkpoint جزو این زمان‌گیری نیست. CPU و CUDA tableها را نباید به‌صورت ساده با هم جمع کرد؛ فعالیت‌ها می‌توانند همپوشانی داشته باشند. شکل واقعی U در recurrence و حافظه peak نیز گزارش می‌شوند.

## ادامه آموزش اصلی بدون شروع مجدد

پس از پایان profiler:

```bash
RESUME_CHECKPOINT=outputs/reference_full_b16_01/last.pt \
TRAINING_OUTPUT=outputs/reference_full_b16_01 \
BATCH_SIZE=16 EPOCHS=1 VALIDATION_BATCHES=32 \
VALIDATE_EVERY_BATCHES=125 CHECKPOINT_EVERY_BATCHES=25 \
bash scripts/train_reference_full.sh
```

داده، batch و تنظیمات اصلی ثابت می‌مانند. خروجی این ابزار برای تصمیم درباره CPU، GPU، batch یا بهینه‌سازی recurrence است؛ کاهش patch، حذف sidechain یا برگشت به backward بومی برای سریع‌ترشدن، بخشی از این ابزار نیست.

تست‌های محلی ابزار، انتخاب pairهای بعدی و حفظ مشتق مختلط هنگام افزودن scope را کنترل می‌کنند. اندازه‌گیری SRU واقعی روی CUDA باید روی سرور انجام شود.
