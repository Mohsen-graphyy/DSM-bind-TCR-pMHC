# اجرای pilot اصلاح‌شده و مقایسه با ATLAS

این راهنما برای همان سروری است که آزمایش گرادیان و ATLAS قبلی را روی آن اجرا کردید. محیط Python/CUDA فعلی را نگه دارید؛ نیازی به اجرای دوباره setup نیست. آدرس و پورت SSH زیر مربوط به اتصال قبلی است؛ اگر instance عوض شده، مقادیر جدید Vast را جایگزین کنید.

## هدف و تغییرات

در آزمایش گرادیان محیط شما، backend بومی SRU گرادیان مختلط لازم برای آموزش بخش‌هایی از مدل، از جمله headهای انرژی، را برنمی‌گرداند. مسیر `reference` از recurrence خود SRU با عملیات PyTorch استفاده می‌کند. در تست تشخیصی، انرژی و نیرو با مسیر بومی توافق داشتند و مشتق پارامتری reference با finite difference سازگار بود. آموزش کامل این مسیر باید روی GPU شما هم اجرا و بررسی شود.

آموزش جدید از `DSMBind/ckpts/model.skempi.allatom` شروع می‌شود؛ وزن fine-tune قبلی ورودی این آموزش نیست. معماری، داده‌های chain-pair، patchها و loss اصلی شامل sidechain حفظ می‌شوند. این اصلاح، مدل را به یک مدل همزمانِ کل کمپلکس چندزنجیره‌ای تبدیل نمی‌کند.

برای کنترل هزینه، بسته‌ای از داده‌های اصلی شما آماده شده است: ۱۲۸ pair از train و ۳۲ pair از validation، با نمونه‌گیری ثابت و بدون تغییر مختصات. فایل `data/reference_pilot/bundle.zip` حدود ۳٫۲ مگابایت است. ATLAS در آموزش یا انتخاب checkpoint استفاده نمی‌شود. این یک pilot تشخیصی است؛ جایگزین آموزش نهایی روی کل داده‌ها نیست.

## ۱. روی سیستم خودتان: ارسال کد به GitHub

در PowerShell و از ریشه همین پروژه:

```powershell
git add .gitignore DSMBind/bindenergy/apps/tcr_pmhc/train.py DSMBind/tests/test_reference_pilot.py DSMBind/REFERENCE_FINETUNE_FA.md scripts/pack_reference_pilot.py scripts/train_reference_pilot.sh scripts/evaluate_reference_pilot.sh
git commit -m "Add reference SRU training pilot with gradient and validation checks"
git push origin main
```

بسته داده در gitignore است؛ آن را با scp منتقل می‌کنیم. اگر داده‌های آموزش اصلی تغییر نکرده‌اند، بسته موجود آماده است و نیاز به ساخت مجدد ندارد. برای ساخت یک بسته جدید با همان ابزار، مسیر خروجی جدید تعیین کنید:

```powershell
python scripts/pack_reference_pilot.py --output-dir data/reference_pilot_v2/bundle
```

## ۲. روی سرور: دریافت کد و ساخت پوشه

در ترمینال SSH بازِ سرور:

```bash
cd /workspace/DSM-bind-TCR-pMHC
git pull --ff-only origin main
mkdir -p data/reference_pilot
```

## ۳. روی سیستم خودتان: انتقال فقط بسته pilot

در PowerShell محلی، از ریشه پروژه:

```powershell
scp -P 23809 ".\data\reference_pilot\bundle.zip" root@ssh9.vast.ai:/workspace/DSM-bind-TCR-pMHC/data/reference_pilot/
```

وزن اصلی DSMBind، بسته ATLAS و cache امبدینگ باید از اجرای قبلی روی سرور موجود باشند. فایل‌های PDB خام آموزش، Excel و وزن fine-tune قبلی برای این pilot لازم نیستند. اگر سرور جدید است، علاوه بر setup قبلی، بسته ATLAS و وزن اصلی و cache موجود را هم طبق راهنمای ATLAS منتقل کنید.

## ۴. روی سرور: استخراج و آموزش کوتاه

```bash
cd /workspace/DSM-bind-TCR-pMHC
python -m zipfile -e data/reference_pilot/bundle.zip data/reference_pilot
bash scripts/train_reference_pilot.sh
```

اسکریپت ابتدا cache مشترک `data/esm_cache/esm2_t36_3B.sqlite` را بررسی و فقط امبدینگ توالی‌های کمبود را محاسبه می‌کند. اگر فایل وزن ESM هنوز در cache دانلود موجود باشد، دوباره دانلود نمی‌شود. سپس با batch=1، learning rate=1e-5 و دو epoch، حداکثر ۲۵۶ گام آموزش انجام می‌شود. reference می‌تواند از backend بومی کندتر باشد؛ زمان و هزینه دقیق از قبل معلوم نیست. لاگ و هزینه instance را پایش کنید.

پیش از اولین update، مدل اصلی روی همان ۳۲ pair validation و نویز ثابت ارزیابی می‌شود. بعد از هر epoch همان ارزیابی تکرار می‌شود. اگر گرادیان هر head انرژی غایب یا غیرمتناهی باشد، آموزش پیش از optimizer.step متوقف می‌شود. گرادیان‌های غیرمتناهی سایر پارامترها نیز باعث توقف می‌شوند.

اگر ارتباط قطع شد و `last.pt` ساخته شده بود، در همان checkout و با همان داده‌ها ادامه دهید:

```bash
RESUME_CHECKPOINT=outputs/reference_pilot_01/last.pt bash scripts/train_reference_pilot.sh
```

برای اجرای مستقل بعدی، پوشه خروجی جدید لازم است؛ اجرای تازه روی پوشه پر مجاز نیست. checkpoint آموزش قبلی با backend بومی را resume نکنید.

## ۵. روی سرور: بررسی خروجی pilot

```bash
cat outputs/reference_pilot_01/final_metrics.json
cat outputs/reference_pilot_01/head_gradient_checks.json
```

فایل‌های مهم:

- `baseline_validation.json`: loss و اجزای آن پیش از آموزش، همراه hash وزن اولیه؛ `zero_force_loss` مرجع مدل با نیروی صفر است، نه هدفی که صرفاً نزدیک‌شدن به آن موفقیت باشد.
- `validation_epoch_1.json` و `validation_epoch_2.json`: همان جزئیات بعد از آموزش؛ برای تشخیص افت نیرو/فروپاشی بررسی شوند.
- `head_gradient_checks.json`: norm گرادیان headها در سه batch اول؛ نبودن خطا در کل اجرا نیز لازم است.
- `head_updates.json`: میزان تغییر headها نسبت به وزن اصلی در بهترین و آخرین checkpoint.
- `final_metrics.json`: مقایسه best validation با baseline و مشخص‌کردن اینکه best هنوز وزن اصلی است یا آموزش‌دیده.
- `best.pt`: بهترین checkpoint با معیار validation؛ اگر هیچ بهبودی رخ ندهد، همان وزن اصلی باقی می‌ماند.
- `last.pt`: وضعیت ادامه آموزش؛ الزاماً بهترین مدل نیست.
- `outputs/reference_pilot_01.log`: لاگ آموزش، بیرون از پوشه وزن‌ها.

برای ادامه ارزیابی، `best_checkpoint_is_initial` باید `false` باشد و headهای best تغییر کرده باشند. این‌ها شروط فنی اولیه‌اند؛ بهبود validation به‌تنهایی اثبات یادگیری affinity نیست. اگر خطای گرادیان، NaN یا OOM دیدید، اجرا را متوقف نگه دارید و لاگ را برای بررسی ذخیره کنید. اگر هیچ checkpoint بهتر نشد، فعلاً اجرای ATLAS برای مدل «جدید» توجیه ندارد؛ خروجی pilot باید بررسی شود.

## ۶. روی سرور: مقایسه مدل اصلی و best جدید با ATLAS

```bash
bash scripts/evaluate_reference_pilot.sh
```

این دستور ابتدا شروط فنی مرحله قبل را بررسی می‌کند، سپس همان pipeline ATLAS را با `outputs/reference_pilot_01/best.pt` اجرا می‌کند. امبدینگ‌های قبلی استفاده می‌شوند و خروجی در پوشه جدید ذخیره می‌شود:

```text
outputs/atlas_reference_pilot_01/
  pretrained/
  finetuned/
  run_manifest.json
  evaluation/
    report.md
    metrics.json
    rankings.csv
    per_group_metrics.csv
    observation_scores.csv
    unit_scores.csv
```

نتایج قبلی در `outputs/atlas_compare` باقی می‌مانند. ابتدا `evaluation/report.md` را بخوانید. معیار اصلی همان `score_sum` و جهت آن همان higher-is-stronger است؛ رتبه تجربی از `-delta G` محاسبه می‌شود. score مدل واحد kcal/mol ندارد.

برای مقایسه با نتیجه قبلی این موارد را بررسی می‌کنیم:

1. Spearman و pairwise accuracy روی همان گروه‌های peptide و assay_context، همراه تعداد گروه‌های واجد شرایط.
2. اختلاف paired مدل جدید و pretrained و فاصله اطمینان آن؛ نتیجه مثبت کوچک به‌تنهایی کافی نیست.
3. معیار ثانویه `peptide_score_sum`، بدون عوض‌کردن معیار اصلی پس از مشاهده ATLAS.
4. همبستگی score با N14 و R² تقریب شمارشی؛ وابستگی تقریباً کامل قبلی باید دوباره سنجیده شود. کم‌شدن این همبستگی به‌تنهایی موفقیت نیست؛ کیفیت رتبه‌بندی هم باید بررسی شود.

این آزمایش affinity ranking است و عملکرد رسمی IMMREP را اندازه نمی‌گیرد. همپوشانی احتمالی ATLAS با پیکره آموزش هنوز نامعلوم است. ATLAS اکنون یک benchmark توسعه‌ایِ مشاهده‌شده است؛ برای ادعای تعمیم نهایی به ارزیابی مستقل نیاز داریم. checkpoint را با validation خودتان انتخاب کنید و برای بالا بردن عدد ATLAS چندین تنظیم را روی همین benchmark جست‌وجو نکنید.

## ۷. روی سیستم خودتان: دانلود نتایج برای تحلیل

در PowerShell از ریشه پروژه:

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
scp -P 23809 -r root@ssh9.vast.ai:/workspace/DSM-bind-TCR-pMHC/outputs/reference_pilot_01 .\outputs\
scp -P 23809 root@ssh9.vast.ai:/workspace/DSM-bind-TCR-pMHC/outputs/reference_pilot_01.log .\outputs\
scp -P 23809 -r root@ssh9.vast.ai:/workspace/DSM-bind-TCR-pMHC/outputs/atlas_reference_pilot_01 .\outputs\
```

اگر آموزش متوقف شد یا شروط مرحله ۶ برقرار نشد، فقط دو دستور اول را اجرا کنید؛ فایل‌های pilot برای بررسی علت کافی‌اند. بعد از ذخیره خروجی‌های موردنیاز، instance را مطابق برنامه هزینه خودتان متوقف کنید. CPU این سیستم برای تحلیل فایل‌های گزارش کافی است؛ اجرای آموزش و تولید score روی سرور انجام می‌شود.

## حدود بررسی محلی

تست‌های CPU شامل محافظ گرادیان، انتخاب checkpoint نسبت به baseline با مدل آزمایشی، حفظ داده هنگام بسته‌بندی، و تست‌های موجود benchmark و resume هستند. تست‌های مدل محلی از SRU جایگزین آزمایشی استفاده می‌کنند؛ صحت آموزش کامل SRU واقعی روی GPU با این تست‌ها ادعا نمی‌شود. مرحله ۴ تأیید اجرایی موردنیاز روی محیط شماست.
