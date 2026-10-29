# بررسی کوتاه گرادیان DSMBind روی Vast

هدف این مرحله تشخیص است: آیا مشتق موردنیاز DSM از SRU عبور می‌کند؟ هیچ training، optimizer step، دانلود ESM یا تغییر فایل checkpoint انجام نمی‌شود. برای جلوگیری از هزینه اضافه، قبل از دیدن نتیجه، fine-tune دیگری شروع نکنید.

## ۱. کد را روی ویندوز commit و push کنید

از ریشه پروژه:

```powershell
git add -- DSMBind/bindenergy/models/energy.py DSMBind/bindenergy/apps/tcr_pmhc/gradient_check.py DSMBind/tests/test_gradient_check.py DSMBind/tests/test_model_device.py scripts/check_dsmbind_gradients.py scripts/check_dsmbind_gradients.sh DSMBind/GRADIENT_CHECK_FA.md
git diff --cached --stat
git commit -m "Add fixed-noise mixed-gradient diagnostic against SRU reference"
git push origin main
```

## ۲. در SSH همان سرور که benchmark Atlas اجرا شد

```bash
cd /workspace/DSM-bind-TCR-pMHC
git pull --ff-only
ls -lh outputs/atlas_compare/dataset.sqlite
ls -lh data/esm_cache/esm2_t36_3B.sqlite
ls -lh artifacts/checkpoints/finetuned_best.pt
```

این‌ها همان فایل‌های اجرای قبلی هستند. نیاز به upload داده یا ساخت embedding جدید نیست. اگر مسیرهای قبلی متفاوت بوده‌اند، در مرحله بعد متغیر مسیر مربوطه را تنظیم کنید.

## ۳. تست کوتاه را اجرا کنید

```bash
bash scripts/check_dsmbind_gradients.sh
```

پیش‌فرض: یک pair واقعی، چهار residue از هر partner، هر دو checkpoint، دو پارامتر head و دو اندازه perturbation. crop فقط در RAM است و dataset اصلی یا نتیجه Atlas را تغییر نمی‌دهد. این آزمون کوچک، گام تشخیصی اولیه است و تأیید کامل داده‌های پنج‌زنجیره‌ای محسوب نمی‌شود.

دو backend مقایسه می‌شوند:

- `native`: مسیر فعلی SRU/CUDA؛
- `reference`: همان معماری و همان وزن‌ها؛ فقط recurrence با تابع `elementwise_recurrence_naive` از SRU نصب‌شده، با عملیات قابل‌مشتق‌گیری PyTorch، به‌طور موقت جایگزین می‌شود. پس از خروج از context روش اصلی برگردانده می‌شود. dependencyها و فایل SRU تغییر نمی‌کنند.

مدل در `eval` است تا dropout خاموش باشد، اما autograd روشن می‌ماند. seed همه مولدهای Python/NumPy/PyTorch برای هر بار محاسبه بازپخش می‌شود؛ پس نویز در دو سمت finite difference یکسان است.

پیش‌فرض، مؤلفه **translation** از همان DSM objective است. این مؤلفه نیز به mixed second derivative نیاز دارد و در آزمون کوتاه نیازی به محاسبه torque/inertia ندارد. augmentation چرخش backbone و side chain همچنان همان augmentation مدل است. mode کامل با `--component full` در CLI Python موجود است، ولی برای شروع لازم نیست.

برای جلوگیری از cancellation عددی، ثابت مستقل از وزنِ noise به صورت جبری از loss حذف می‌شود:

```text
||predicted - noise_target||²
= (||predicted||² - 2 predicted·noise_target) + ||noise_target||²
```

مشتق قسمت اول همان مشتق loss کامل است؛ فقط سیگنال کوچک گرادیان در عدد بزرگ noise گم نمی‌شود. برابری این رابطه و unchanged بودن forward عادی در تست CPU بررسی شده است.

reference به علت حلقه PyTorch از CUDA kernel کندتر است؛ چهار residue برای محدودکردن هزینه انتخاب شده‌اند. زمان قطعی روی GPU اعلام نمی‌شود. در خروجی ترمینال checkpoint، backend و نتیجه هر probe چاپ می‌شود. اگر حافظه کم بود یا اجرا بیش‌ازحد طولانی شد، متوقف کنید؛ وزن‌ها و cache عوض نمی‌شوند.

## ۴. چه خروجی‌هایی داریم؟

```text
outputs/gradient_check_01/
  report.md                  جدول وضعیت تست‌ها
  diagnostic.json            مقادیر loss/force، مقایسه backendها و خطاهای احتمالی
  parameter_gradients.csv     وجود و اندازه گرادیان برای تمام پارامترها
  finite_differences.json     مشتق خودکار و numerical برای probeها
```

در `parameter_gradients.csv`، `gradient_present=False` با گرادیان صفر فرق دارد. برای هر probe، جایگاه حساس‌ترین مؤلفه گرادیان reference انتخاب می‌شود و همان مؤلفه در native نیز آزموده می‌شود؛ پارامترها فقط در RAM تغییر جزئی می‌کنند و حتی در صورت exception به مقدار اصلی برمی‌گردند.

## ۵. نتیجه را چطور بخوانیم؟

| نتیجه | تفسیر و اقدام |
|---|---|
| reference=`pass`، native=`fail_missing_gradient` یا `fail_gradient_mismatch`، همراه با agreement انرژی و force | شاهد مستقیم مشکل mixed-gradient در مسیر native؛ آموزش طولانی با این backend را ادامه ندهید. قدم بعد، pilot بسیار کوتاه با recurrence قابل‌مشتق‌گیری است. |
| هر دو `pass` | این probeهای کوچک سالم‌اند؛ هنوز کل training تأیید نشده. بررسی بعدی مؤلفه‌های کامل loss و کیفیت/وزن‌دهی pairهاست. |
| `inconclusive_low_signal` | سیگنال آن‌قدر کوچک است که FD قابل اتکا نیست؛ این وضعیت pass نیست. نتیجه pretrained و یک pair دیگر را بررسی کنید. |
| `inconclusive_fd_unstable` | مشتق با epsilon تغییر می‌کند؛ قبل از نتیجه‌گیری عددی باید pair/epsilon دیگری بررسی شود. |
| reference خطا می‌دهد یا FD آن fail می‌شود | هنوز backend جایگزین هم تأیید نشده؛ training تازه شروع نکنید. |
| انرژی یا بردار force بین backendها agreement ندارد | تفاوت فقط به double backward نسبت داده نمی‌شود؛ ابتدا تفاوت forward/first derivative بررسی شود. |

در `diagnostic.json` زیر `models -> checkpoint -> backend_value_agreement` وضعیت `close` را بخوانید. برای حالت translation بردار force هم مقایسه می‌شود. معیار پیش‌فرض FD: `atol=1e-9` و `rtol=0.2` روی signal loss؛ این‌ها در گزارش ثبت می‌شوند. threshold را صرفاً برای گرفتن pass تغییر ندهید.

مقادیر checkpoint fine-tuned ممکن است به دلیل saturation سیگنال بسیار کمی داشته باشند؛ برای تشخیص مسیر گرادیان، نتیجه pretrained اهمیت ویژه دارد. خطا/سیگنال کم به‌تنهایی علت ضعیف‌شدن ranking را ثابت نمی‌کند.

اگر یک pair نتیجه نداد، فقط یک بار با خروجی جدید تکرار کنید:

```bash
PAIR_INDEX=1 DIAG_OUTPUT=outputs/gradient_check_02 bash scripts/check_dsmbind_gradients.sh
```

از بازنویسی خروجی قبلی جلوگیری می‌شود. اگر cache یا DB شما جای دیگری است:

```bash
DATASET=/path/dataset.sqlite ESM_CACHE=/path/esm2_t36_3B.sqlite bash scripts/check_dsmbind_gradients.sh
```

## ۶. نتیجه را دانلود کنید

در PowerShell ویندوز (نه داخل SSH)، با همان سرور قبلی:

```powershell
scp -P 23809 -r root@ssh9.vast.ai:/workspace/DSM-bind-TCR-pMHC/outputs/gradient_check_01 .\
```

اگر instance/port شما عوض شده، مشخصات فعلی Vast را جایگزین کنید. پوشه فقط گزارش‌های کوچک است؛ وزن یا embedding را دانلود مجدد نکنید. چهار فایل نتیجه را برای تحلیل نگه دارید. تا مشخص‌شدن نتیجه لازم نیست GPU روشن بماند.

## محدودیت اعتبارسنجی محلی

CPU tests با یک تابع عمداً معیوب نشان می‌دهند ابزار، گرادیان مرتبه دوم قطع‌شده را از مشتق درست جدا می‌کند. این آزمون، تأیید SRU واقعی روی CUDA نیست. CPU smokeهای مدل از FakeSRU استفاده می‌کنند؛ اجرای این دستور روی Vast آزمون واقعی بعدی است. تغییر فعلی مشکل قبلی را از پیش حل‌شده فرض نمی‌کند.

منابع فنی:

- SRU reference: https://github.com/asappresearch/sru/blob/c2d44e62b90115db59ab92ccea4b2e5b77077cf9/sru/ops.py
- PyTorch double backward: https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
