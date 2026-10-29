# آموزش کامل اصلاح‌شده DSMBind

این اجرا آزمایش مجددِ معتبرِ آموزش روی داده‌های اصلی است؛ بهبود affinity یا امتیاز مسابقه را تضمین نمی‌کند. pilot نشان داد headهای انرژی با مسیر reference گرادیان می‌گیرند و آپدیت می‌شوند. افت ATLAS در pilot به‌تنهایی دلیل بی‌فایده‌بودن آموزش کامل نیست. با تأمین زمان و بودجه، آموزش کامل یک آزمایش موجه است.

## داده و تنظیمات

- train: ۳۴٬۰۹۳ ساختار و ۱۳۷٬۸۶۳ pair؛ همه pairهای train در هر epoch استفاده می‌شوند.
- validation: ۳٬۱۱۷ ساختار و ۱۲٬۵۳۲ pair؛ برای انتخاب وزن، ۵۱۲ pair با seed و نویز ثابت استفاده می‌شوند. این عدد subset اعتبارسنجی است؛ داده آموزشی محدود نمی‌شود.
- test و final_unseen وارد آموزش یا انتخاب checkpoint نمی‌شوند.
- شروع مجدد از `DSMBind/ckpts/model.skempi.allatom`؛ نه وزن قدیمی و نه best pilot.
- reference SRU، batch=1، loss کامل با sidechain، learning rate=1e-6 و سه epoch. learning rate کمتر، انتخابی محافظه‌کارانه برای اجرای بسیار طولانی‌تر است؛ مقدار بهینه اثبات نشده است.
- ارزیابی و امکان انتخاب best هر ۲۰۰۰ update و انتهای epoch؛ ذخیره last هر ۵۰۰ update. سه epoch حداکثر ۴۱۳٬۵۸۹ update دارد؛ زمان دقیق را از سرعت واقعی لاگ برآورد کنید.
- گرادیان headهای انرژی و متناهی‌بودن گرادیان‌ها کنترل می‌شود. اگر هیچ وزن آموزش‌دیده‌ای از baseline validation بهتر نشود، best همان وزن اولیه می‌ماند و این وضعیت در final_metrics ثبت می‌شود.

## ۱. روی سرور، دریافت نسخه جدید

در همان محیط GPU که pilot را با موفقیت اجرا کردید:

```bash
cd /workspace/DSM-bind-TCR-pMHC
git pull --ff-only origin main
mkdir -p data/dsmbind_tcr_pmhc
ls -lh data/dsmbind_tcr_pmhc/train.sqlite data/dsmbind_tcr_pmhc/validation.sqlite
```

نیازی به نصب مجدد محیط نیست. فایل‌های آماده‌شده اصلی احتمالاً از اجرای اول موجودند. اگر نیستند، مرحله ۲ را انجام دهید. داده خام PDB و Excel برای این اجرای مبتنی بر DB لازم نیستند.

## ۲. فقط اگر DBهای اصلی روی سرور نیستند

در PowerShell کامپیوتر خودتان، از ریشه پروژه؛ آدرس و پورت را اگر instance عوض شده جایگزین کنید:

```powershell
scp -P 23809 ".\data\dsmbind_tcr_pmhc\train.sqlite" root@ssh9.vast.ai:/workspace/DSM-bind-TCR-pMHC/data/dsmbind_tcr_pmhc/
scp -P 23809 ".\data\dsmbind_tcr_pmhc\validation.sqlite" root@ssh9.vast.ai:/workspace/DSM-bind-TCR-pMHC/data/dsmbind_tcr_pmhc/
```

SQLite باید هنگام کپی بسته باشد. train حدود ۱٫۶ گیگابایت و validation حدود ۱۳۸ مگابایت است. DBها در Git نیستند و با scp منتقل می‌شوند.

## ۳. روی سرور، اجرای کامل

بهتر است فرمان را در نشست ماندگاری مثل tmux که روی سرور موجود است اجرا کنید تا قطع SSH آموزش را نبندد:

```bash
cd /workspace/DSM-bind-TCR-pMHC
bash scripts/train_reference_full.sh
```

ابتدا cache امبدینگ train و validation تکمیل می‌شود. cache قبلی استفاده می‌شود؛ فقط توالی‌های کمبود محاسبه می‌شوند. مسیر پیش‌فرض `data/esm_cache/esm2_t36_3B.sqlite` است. اگر cache کامل آموزش اولیه جای دیگری است، مسیر آن را بدهید تا کار تکرار نشود:

```bash
ESM_CACHE=/absolute/path/to/existing.sqlite bash scripts/train_reference_full.sh
```

خروجی مستقل در `outputs/reference_full_01` ذخیره می‌شود. ابتدا baseline مدل اصلی روی validation محاسبه می‌شود و سپس آموزش شروع می‌شود. بهبود معیار DSM، تضمین بهبود affinity نیست؛ عملکرد ATLAS بعد از انتخاب best سنجیده می‌شود.

## ۴. مشاهده روند و ادامه پس از قطع

در ترمینال دوم سرور:

```bash
tail -f outputs/reference_full_01.log
```

فایل `outputs/reference_full_01/validation_history.json` جزئیات ارزیابی‌های بین epochها، force RMS و مرجع نیروی صفر را نگه می‌دارد. `best.pt` و `last.pt` ممکن است متفاوت باشند. اگر اجرا قطع شد، در همان پوشه و با همان تنظیمات ادامه دهید:

```bash
RESUME_CHECKPOINT=outputs/reference_full_01/last.pt bash scripts/train_reference_full.sh
```

اگر ESM_CACHE غیرپیش‌فرض استفاده کردید، آن متغیر را هنگام resume نیز بدهید. ترتیب pairها، optimizer، scheduler و RNG از checkpoint بازیابی می‌شوند. DBهای train/validation و وزن اولیه با hash کنترل می‌شوند. تغییر seed، batch یا subset اعتبارسنجی هنگام ادامه مجاز نیست.

## ۵. پس از پایان، ATLAS

```bash
cat outputs/reference_full_01/final_metrics.json
PILOT_OUTPUT=outputs/reference_full_01 ATLAS_WORK_DIR=outputs/atlas_reference_full_01 bash scripts/evaluate_reference_pilot.sh
```

نام اسکریپت ارزیابی از pilot باقی مانده است، اما مسیر بالا وزن آموزش کامل را به آن می‌دهد. اسکریپت اگر best هنوز وزن اولیه باشد یا headها تغییر نکرده باشند، متوقف می‌شود. مدل اصلی و best آموزش کامل روی همان ATLAS مقایسه می‌شوند؛ جهت و aggregation بر اساس نتایج عوض نمی‌شوند. گزارش در `outputs/atlas_reference_full_01/evaluation/report.md` است. این آزمایش معیار رسمی مسابقه نیست.

## ۶. دانلود خروجی‌ها

روی کامپیوتر خودتان:

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
scp -P 23809 -r root@ssh9.vast.ai:/workspace/DSM-bind-TCR-pMHC/outputs/reference_full_01 .\outputs\
scp -P 23809 root@ssh9.vast.ai:/workspace/DSM-bind-TCR-pMHC/outputs/reference_full_01.log .\outputs\
scp -P 23809 -r root@ssh9.vast.ai:/workspace/DSM-bind-TCR-pMHC/outputs/atlas_reference_full_01 .\outputs\
```

وزن‌ها و cache در gitignore باقی می‌مانند. کد و راهنما روی GitHub قرار می‌گیرند. نتایج واقعی GPU این اجرای کامل باید پس از اجرا بررسی شوند؛ تست‌های محلی CPU جایگزین آن نیستند.
