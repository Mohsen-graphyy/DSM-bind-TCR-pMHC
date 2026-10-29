# مقایسه رتبه‌بندی ATLAS با DSMBind اصلی و fine-tuned

این مسیر **inference و ارزیابی** است و هیچ وزنی را آموزش نمی‌دهد یا تغییر نمی‌دهد. امتیاز خام DSMBind واحد kcal/mol ندارد؛ آن را به ΔG تبدیل نمی‌کنیم. ΔG آزمایشگاهی فقط در ارزیابی استفاده می‌شود:

```text
ΔG کوچک‌تر / منفی‌تر = قوی‌تر
DSMBind score بزرگ‌تر = قوی‌تر، مطابق قرارداد کد اصلی
مقایسه رتبه‌ای: correlation(score, -ΔG)
```

اجرای اصلی از **ریشه همین Git checkout**، یعنی پوشه والد `DSMBind` است.

## ۱. چه چیزی از داده شما آماده شده است؟

ورودی بررسی‌شده:

```text
C:\Users\hajmo\OneDrive\Desktop\Tesses\atlas dataset\outputs\atlas_peptide_ranking\atlas_peptide_ranking_index.csv
```

اکسل اصلی تغییر نمی‌کند. فایل index همراه آن همان observationها و جایگاهشان در آرایه‌های اکسل را دارد؛ برای اجرای قابل‌تکرار از CSV استفاده می‌شود. metadata نگاشت زنجیره‌ها و شرایط آزمایش از `atlas_model_test` خوانده می‌شود.

در اجرای CPU محلی، ۲۳۶ observation به ۲۰۶ ساختار و ۲۰۶ واحد زیستی تبدیل شدند. از ساختارها ۸۷۱ pair و ۵۱۱ توالی یکتا با ۴۶٬۷۱۹ residue استخراج شد. embedding خام float16 حدود ۲۲۸ MiB است (پیش از فشرده‌سازی و سربار SQLite). نیازی به انتقال cache قدیمی ۳۳ گیگابایتی نیست.

هر ساختار فقط یک‌بار برای هر checkpoint امتیاز می‌گیرد. observationهای تکراری حفظ می‌شوند و در ارزیابی، ΔG میانه برای هر واحد زیستی محاسبه می‌شود. واحد زیستی با توالی تمام زنجیره‌های TCR و pMHC تعریف می‌شود، نه صرفاً TCR ID یا نام PDB.

نقش زنجیره‌های Atlas با تطبیق دقیق توالی CA تعیین می‌شود. در بسته فعلی، alpha به `B` و beta به `A` نگاشت می‌شوند تا قرارداد داده آموزشی حفظ شود؛ MHC به `M/N` و peptide به `P` می‌روند. این نگاشت بر اساس ترتیب حروف حدس زده نمی‌شود.

برای missing atomها مختصات صفر و mask مدل حفظ می‌شود. policy پیش‌فرض `ca`، CA را الزامی می‌کند و کمبود N/C/O را گزارش می‌دهد. `--backbone-policy strict` وجود N/CA/C را الزامی می‌کند و ممکن است روی چند ساختار فعلی خطا بدهد. قالب غیرمعمول چند ATOM اطلس نیز با parser صریح whitespace خوانده می‌شود؛ مختصات و فایل اصلی تغییر نمی‌کنند.

## ۲. بازسازی بسته روی کامپیوتر خودتان (بدون GPU)

اگر `data/atlas_benchmark/bundle.zip` موجود است، این مرحله قبلاً انجام شده است. برای نسخه جدید داده، مسیر خروجی جدید انتخاب کنید؛ bundle غیرخالی عمداً overwrite نمی‌شود.

```powershell
python scripts/dsmbind_benchmark.py pack-atlas `
  --index "C:\Users\hajmo\OneDrive\Desktop\Tesses\atlas dataset\outputs\atlas_peptide_ranking\atlas_peptide_ranking_index.csv" `
  --atlas-manifest "C:\Users\hajmo\OneDrive\Desktop\Tesses\atlas dataset\outputs\atlas_model_test\model_inputs\manifest.csv" `
  --atlas-root "C:\Users\hajmo\OneDrive\Desktop\Tesses\atlas dataset\outputs\atlas_model_test" `
  --output-dir data/atlas_benchmark/bundle `
  --zip
```

ساختار قابل انتقال:

```text
data/atlas_benchmark/bundle/
  bundle.json          تعدادها، hashها و قرارداد داده
  structures.csv       ورودی مدل، chain map و مسیر نسبی PDB؛ بدون ΔG
  observations.csv     شناسه observation، ΔG و شرایط آزمایش؛ برای ارزیابی
  pdbs/                ساختارهای یکتا
```

CPU preprocessing اختیاری است و روی سرور نیز خودکار انجام می‌شود:

```powershell
python scripts/dsmbind_benchmark.py prepare `
  --bundle data/atlas_benchmark/bundle `
  --work-dir data/atlas_benchmark/prepared
```

روی سیستم محلی Python 3.13 و PyTorch CPU موجود بود و این مرحله اجرا شد. برای فقط `pack` به GPU، SRU یا ESM نیاز ندارید. برای `prepare`، PyTorch CPU و NumPy و وابستگی‌های import داده DSMBind باید نصب باشند. اجرای wrapper از ریشه پروژه نیاز به `pip install -e` محلی ندارد و محدودیت تاریخی Python در pyproject را دستکاری نمی‌کند.

## ۳. کد را به GitHub ببرید

ابتدا تغییرات را ببینید؛ تغییر قبلی `constants.py` و فایل‌های نامرتبط را ناخواسته وارد commit نکنید:

```powershell
git status --short
git diff --stat
git add -- .gitignore scripts/dsmbind_benchmark.py scripts/setup_inference_vast.sh scripts/run_atlas_benchmark.sh DSMBind/bindenergy/apps/tcr_pmhc/benchmark.py DSMBind/bindenergy/apps/tcr_pmhc/benchmark_io.py DSMBind/bindenergy/apps/tcr_pmhc/ranking.py DSMBind/bindenergy/data/tcr_pmhc.py DSMBind/tests/test_structure_benchmark.py DSMBind/examples/structure_manifest.csv DSMBind/README.md DSMBind/ATLAS_BENCHMARK_FA.md reports/atlas_benchmark_readiness_2026-10-29.md
git diff --cached --stat
git commit -m "Add portable structure inference and paired ATLAS ranking benchmark"
git push origin main
```

وزن‌های شخصی، داده Atlas، cacheها، خروجی‌های حجیم و کلیدهای SSH در `.gitignore` قرار دارند. کد و راهنما روی GitHub می‌روند؛ داده و `best.pt` از SCP منتقل می‌شوند. نیازی به push کردن دیتابیس، PDBها یا مدل ESM نیست.

## ۴. انتخاب و آماده‌سازی سرور Vast

یک GPU با ۲۴GB VRAM مانند RTX 3090/4090 و محیط **PyTorch CUDA devel دارای nvcc** انتخاب کنید. فضای دیسک باید برای محیط Python و دانلود checkpoint حدود ۱۲GB مدل ESM-2 و cache کافی باشد؛ حداقل ۳۰GB فضای آزاد عملی‌تر است. قیمت ساعتی و هزینه دیسک/انتقال را در پنل فعلی Vast بررسی کنید؛ این راهنما نرخ ثابتی فرض نمی‌کند.

برای سرور Python 3.10 تا 3.12 انتخاب کنید؛ NumPy<2 این stack برای Python 3.13 مناسب نیست. Python جدیدتر سیستم محلی برای ساخت بسته مشکلی ندارد.

اگر همان محیط قبلی شما سالم است، از آن استفاده کنید. برای محیط تازه، ابتدا در SSH:

```bash
cd /workspace
git clone https://github.com/Mohsen-graphyy/DSM-bind-TCR-pMHC.git
cd DSM-bind-TCR-pMHC
mkdir -p data/atlas_benchmark artifacts/checkpoints
```

اگر checkout از قبل وجود دارد:

```bash
cd /workspace/DSM-bind-TCR-pMHC
git pull --ff-only
mkdir -p data/atlas_benchmark artifacts/checkpoints
```

## ۵. فقط این دو فایل را به سرور منتقل کنید

در PowerShell کامپیوتر خودتان، `PORT` و `HOST` را از Vast جایگزین کنید:

```powershell
scp -P PORT ".\data\atlas_benchmark\bundle.zip" root@HOST:/workspace/DSM-bind-TCR-pMHC/data/atlas_benchmark/
scp -P PORT ".\artifacts\checkpoints\finetuned_best.pt" root@HOST:/workspace/DSM-bind-TCR-pMHC/artifacts/checkpoints/
```

`best.pt` آموزش کامل را منتقل کنید؛ فایل `best (1).pt` مربوط به pilot است. checkpoint pretrained در Git checkout، مسیر `DSMBind/ckpts/model.skempi.allatom` موجود است. اندازه/وجود آن را پس از clone بررسی کنید.

نسخه بررسی‌شده `Downloads/best.pt` قبلاً در مسیر محلی `artifacts/checkpoints/finetuned_best.pt` کپی شده است؛ هر دو فایل آماده انتقال‌اند و در Git قرار نمی‌گیرند.

روی سرور:

```bash
cd /workspace/DSM-bind-TCR-pMHC
python -m zipfile -e data/atlas_benchmark/bundle.zip data/atlas_benchmark
sha256sum artifacts/checkpoints/finetuned_best.pt
ls -lh DSMBind/ckpts/model.skempi.allatom
```

SHA-256 نسخه fine-tuned بررسی‌شده:

```text
305002a101fe2465324edb695034239ee9a6aa81e52f807c12a5a39fb2af3017
```

اگر وزن دیگری را عمداً آزمایش می‌کنید، hash و نام آن باید در نتیجه ثبت شود؛ runner این کار را خودکار انجام می‌دهد.

## ۶. وابستگی inference را نصب کنید

```bash
bash scripts/setup_inference_vast.sh
```

این اسکریپت PyTorch موجود در محیط Vast را نگه می‌دارد، SRU را در صورت نبودن نصب می‌کند و اصلاح `.type()` به `.scalar_type()` را برای سازگاری کامپایل انجام می‌دهد. backup فایل CUDA نگه داشته می‌شود. از `importlib` برای یافتن فایل استفاده می‌کند تا قبل از patch، کامپایل ناموفق شروع نشود.

pyproject اصلی Python <3.11 را درخواست می‌کند. این اسکریپت برای نصب inference در محیط جدید از `--ignore-requires-python --no-deps` استفاده می‌کند؛ این کار تأیید صحت double-backward آموزش نیست. کار فعلی فقط `predict` است و قابلیت اجرای واقعی GPU باید روی سرور شما تأیید شود. در این تغییر، GPU واقعی محلی در دسترس نبود.

## ۷. پیش‌بررسی کم‌هزینه قبل از ESM

```bash
python scripts/dsmbind_benchmark.py run \
  --bundle data/atlas_benchmark/bundle \
  --work-dir outputs/atlas_compare \
  --pretrained DSMBind/ckpts/model.skempi.allatom \
  --finetuned artifacts/checkpoints/finetuned_best.pt \
  --embeddings data/esm_cache/esm2_t36_3B.sqlite \
  --preparation-only
```

این دستور checkpointها، معماری، داده و workload را روی CPU بررسی می‌کند و ESM/GPU inference را شروع نمی‌کند. انتظار داده فعلی: ۲۰۶ ساختار، ۸۷۱ pair و ۵۱۱ توالی. اگر شمارش تفاوت دارد، مسیر ورودی را بررسی کنید.

## ۸. اجرای کامل با یک دستور

```bash
bash scripts/run_atlas_benchmark.sh
```

ترتیب خودکار:

1. کنترل checksum و preprocessing همان ساختارها برای هر دو checkpoint؛
2. بررسی cache و محاسبه فقط embeddingهای مفقود ESM-2، یک‌بار برای هر توالی؛
3. inference با مدل pretrained؛
4. inference با `finetuned_best.pt`؛
5. اتصال scoreها به observationها با شناسه؛
6. ساخت رتبه‌بندی و گزارش مقایسه‌ای.

اگر SSH ممکن است قطع شود، دستور را در `tmux` اجرا کنید. اگر برنامه قطع شد، **همان دستور** را دوباره اجرا کنید: embeddingهای commitشده و مرحله امتیازدهی کامل‌شده دوباره محاسبه نمی‌شوند. مرحله scoring نیمه‌تمام از ابتدا تکرار می‌شود، اما فقط روی ۲۰۶ ساختار است. تغییر checkpoint/config یا دستکاری خروجی کامل‌شده عمداً cache را نامعتبر می‌کند؛ برای آزمایش متفاوت `WORK_DIR` جدید بدهید.

```bash
WORK_DIR=outputs/atlas_compare_v2 BATCH_SIZE=2 ESM_TOKEN_BUDGET=256 bash scripts/run_atlas_benchmark.sh
```

اگر CUDA OOM هنگام ESM رخ داد، token budget را کم کنید. اگر هنگام scoring رخ داد، batch size را کم کنید. کاهش batch size برای خروجی کامل‌شده نیازمند work directory جدید است. مدل ESM باید همان 3B و لایه ۳۶ بماند؛ embedding کوچک‌تر با وزن‌ها سازگار نیست.

برای cache قدیمی موجود، بدون انتقال دوباره آن:

```bash
ESM_CACHE=/path/to/existing/esm2_t36_3B.sqlite bash scripts/run_atlas_benchmark.sh
```

زمان کل شامل دانلود مدل ESM، کامپایل اولیه SRU و inference است؛ زمان ثابت یا هزینه قطعی وعده داده نمی‌شود. برای بودجه حدود ۲۰ دلار، ابتدا preflight و log دانلود/کامپایل را بررسی کنید و پس از ذخیره خروجی، GPU را متوقف/خاموش کنید. برای حذف instance ابتدا checkpoint، report و cache موردنیاز را دانلود کنید.

## ۹. خروجی‌ها و معنی نتیجه

```text
outputs/atlas_compare/
  preparation_report.json
  missing_backbone_atoms.csv
  geometry_pairs.csv
  dataset.sqlite
  pretrained/{pair_scores.csv,complex_scores.csv,completion.json,...}
  finetuned/{pair_scores.csv,complex_scores.csv,completion.json,...}
  logs/{embeddings.log,pretrained.log,finetuned.log}
  run_manifest.json
  evaluation/
    report.md
    metrics.json
    observation_scores.csv
    unit_scores.csv
    rankings.csv
    per_group_metrics.csv
```

معیار اصلی از پیش تعیین‌شده `score_sum` است: مجموع symmetric score همه pairهای retained. جهت score با label انتخاب نمی‌شود. معیار ثانویه `peptide_score_sum` فقط pairهای TCR–peptide را جمع می‌کند. `score_mean` و `score_per_contact` هم برای تشخیص وابستگی به اندازه interface گزارش می‌شوند؛ بهترین مورد را بعد از دیدن test به‌عنوان «معیار اصلی» معرفی نکنید.

- `rankings.csv`: در هر پپتید، رتبه واقعی از ΔG و رتبه هر مدل؛ رتبه ۱ یعنی قوی‌ترین. ties رتبه متوسط می‌گیرند.
- `observation_scores.csv`: تمام ۲۳۶ measurement اولیه با ΔG و score هر دو مدل؛ نمونه‌های تکراری حذف نمی‌شوند.
- `unit_scores.csv`: واحدهای مستقل‌تر برای محاسبه معیارها؛ ΔG میانه measurementهای همان واحد و score میانگین ساختارهای یکتای آن.
- `per_group_metrics.csv`: Spearman، Kendall tau-b، pairwise accuracy و top1 credit؛ eligibility همه گروه‌ها ثبت می‌شود.
- `metrics.json`: میانگین macro گروه‌ها، اختلاف paired روی دقیقاً گروه‌های مشترک و bootstrap توصیفی گروهی. CI با کمتر از دو گروه محاسبه نمی‌شود؛ این CI استقلال templateها/TCRها را تضمین نمی‌کند.
- `geometry_pairs.csv`: baseline شمارش جفت‌اتم‌های زیر ۱۴Å؛ گزارش میزان شباهت score به این baseline را هم می‌سنجد.
- `run_manifest.json` و `completion.json`: hash وزن‌ها، کد، داده، subset امبدینگ استفاده‌شده، نسخه کتابخانه‌ها و Git commit.

گروه‌های کمتر از سه واحد برای correlation رتبه‌ای معتبر گزارش نمی‌شوند. برای گروه دو عضوی با ΔG غیرمساوی، مقایسه دوتایی همچنان گزارش می‌شود؛ prediction مساوی نیم امتیاز دارد. singleton را نمی‌توان rank benchmark دانست. به تعداد گروه‌های قابل ارزیابی توجه کنید، نه فقط ۲۳۶ observation.

در بسته فعلی ۱۲۹ گروه peptide وجود دارد؛ فقط ۱۹ گروه با مجموع ۷۵ واحد زیستی حداقل سه عضو دارند. پیش از بررسی ثابت‌بودن label یا prediction، ۷ گروه assay-context حداقل سه عضو دارند. تعداد نهایی گروه‌های eligible در report پس از inference مشخص می‌شود.

مقایسه اصلی مطابق خواسته شما به‌تفکیک peptide است. تحلیل دوم `assay_context`، peptide/HLA/PMID/temperature/assay را ثابت نگه می‌دارد. گروه‌بندی peptide-only ممکن است زمینه‌های آزمایش یا HLA متفاوت را مخلوط کند؛ از آن نتیجه دقیق بیوفیزیکی نگیرید.

این benchmark، affinity ranking اطلس است؛ معیار رسمی IMMREP26 یا تشخیص فعال‌شدن T cell نیست. همپوشانی Atlas با آموزش اصلی یا fine-tune هنوز مستقل بررسی نشده و گزارش آن را unknown می‌داند. مشکل بالقوه گرادیان/فروپاشی score قبلی با اجرای این benchmark «حل» نمی‌شود؛ این آزمایش امکان اندازه‌گیری عملکرد واقعی دو checkpoint را می‌دهد. RMSE بین raw score و kcal/mol محاسبه نمی‌شود.

مقاله مرجع اطلس: https://pmc.ncbi.nlm.nih.gov/articles/PMC5860664/

## ۱۰. دانلود نتیجه و بستن سرور

روی سرور، از ریشه checkout:

```bash
python -m tarfile -c atlas_comparison_results.tar.gz outputs/atlas_compare
```

در PowerShell سیستم خودتان:

```powershell
scp -P PORT root@HOST:/workspace/DSM-bind-TCR-pMHC/atlas_comparison_results.tar.gz .
scp -P PORT root@HOST:/workspace/DSM-bind-TCR-pMHC/data/esm_cache/esm2_t36_3B.sqlite .\data\esm_cache\
```

قبل از دانلود cache، اجرای embedding باید تمام شده و SQLite بسته باشد؛ اسکریپت WAL را checkpoint می‌کند. مسیر مقصد `data/esm_cache` را اگر موجود نیست با `New-Item -ItemType Directory -Force data/esm_cache` بسازید. cache حدود ۲۲۸MiB خام این مجموعه را می‌توان برای آزمایش‌های بعدی نگه داشت. پوشه خروجی شامل DB آماده است؛ اگر فقط report می‌خواهید، `evaluation`، `run_manifest.json` و پوشه‌های score را نگه دارید.

ارزیابی مجدد روی CPU، بدون SRU/GPU/دانلود ESM:

```powershell
python scripts/dsmbind_benchmark.py evaluate `
  --bundle data/atlas_benchmark/bundle `
  --work-dir outputs/atlas_compare `
  --output-dir outputs/atlas_compare/evaluation
```

## ۱۱. inference روی داده جدید با همین pipeline

از `DSMBind/examples/structure_manifest.csv` الگو بگیرید. هر سطر یک observation است:

- `observation_id`: شناسه یکتا؛
- `pdb_path`: مسیر PDB نسبت به CSV یا absolute؛
- `chain_map`: JSON صریح از نام زنجیره اصلی به نقش canonical؛ A=TCR beta، B=TCR alpha، M=MHC alpha، N=MHC beta اختیاری، P=peptide؛
- `peptide`: توالی peptide دقیقاً مطابق PDB؛
- `delta_g_kcal_mol`: اختیاری؛ برای inference بدون label خالی بماند؛
- `strict_group_id`: اختیاری؛ برای مقایسه در شرایط آزمایش همسان.

این pipeline PDB آماده را می‌خواند؛ sequence-only input را به ساختار تبدیل نمی‌کند، mutation را روی WT جعل نمی‌کند و شماره chain را حدس نمی‌زند.

```bash
python scripts/dsmbind_benchmark.py pack \
  --manifest my_new_data/manifest.csv --output-dir data/new_bundle --zip

python scripts/dsmbind_benchmark.py infer \
  --bundle data/new_bundle \
  --work-dir outputs/new_inference \
  --checkpoint artifacts/checkpoints/finetuned_best.pt \
  --embeddings data/esm_cache/esm2_t36_3B.sqlite \
  --device cuda
```

خروجی تک‌مدل `outputs/new_inference/predictions.csv` است. cache موجود گسترش می‌یابد و فقط sequenceهای جدید محاسبه می‌شوند. برای مقایسه دو مدل روی داده جدید، به‌جای `infer` از `run` با `--pretrained` و `--finetuned` استفاده کنید.
