# Thesis review — 22200034_Thesis_EN.docx

**Source:** the file you attached (2,695,727 bytes, modified 02:01) — 733 paragraphs, 46 tables, 25 figures
**Output:** `22200034_Thesis_EN_edited.docx` — **93 edits, all inside body-text sentences**

## Integrity — verified, not assumed

| Check | Result |
|---|---|
| Runs / run-properties / bold / italic | 3617 / 4006 / 477 / 195 — **identical to your file** |
| Headings, chapter titles, TOC entries | **0 changed** |
| Table and figure captions | **0 changed** |
| Cover pages, thesis title (EN and VI) | **0 changed** |
| Runs with altered character formatting | **0** |
| Tables / images / equations | 46 / 25 / 31 — unchanged |
| XML parts changed by my edits | **only `word/document.xml`** — styles, numbering, fonts, headers, footers, images untouched |
| Paragraphs with altered wording | 76 |

---

## 1. What this revision needs to know

Between the two copies on your machine, someone ran a paraphrasing pass over the whole document. It removed some genuine problems, but it **introduced about 30 new grammatical errors**, deleted content in two places, and — ironically — added 16 em-dashes, which is the single most recognisable marker of machine-written English.

The most serious one is on the first page of Chapter 1:

> **Was:** "This thesis aims to **Algorithm-hardware co-design of a hardware-efficient quantized 1D CNN accelerator for ecg arrhythmia classification on FPGA** , covering the full path…"

The thesis title had been pasted in where the verb phrase belonged, leaving the sentence with no predicate, a space before the comma, and `ecg` in lower case. Now:

> **Is:** "This thesis presents an algorithm-hardware co-design of a hardware-efficient quantized 1D-CNN accelerator for ECG arrhythmia classification on FPGA, covering the full path…"

Two places had lost content outright:

- **§1.4** — "– Programmable logic: 41,910 ALMs" ended mid-sentence. `(≈110,000 logic elements) and ≈166,000 registers` had been deleted. Restored.
- **§3.2.5** — the sentence "Figure 3.10 shows the dataflow." had been deleted, which left Figure 3.10 with no citation anywhere. Restored.

---

## 2. Grammar and wording (38 fixed)

Errors that would stop a reader mid-sentence:

| Where | Was | Now |
|---|---|---|
| §3.2.4 | "a single unified finite state **machinethat** controls" | "…state machine that controls" |
| §3.3.1 | "a thin wrapper of only two components **are** the bus converter and the computational core, **which plus** the input memory" | "…of only two components, the bus converter and the computational core, plus the input memory" |
| §4.6 | "The difference is architectural **which include** eight processing units" | "The difference is architectural: eight processing units" |
| §5.1 | "with a maximum deviation of 0 **that enough to** assert exact computation" | "…of 0. Agreement with the software model over the complete test set, reported in §4.4.2, confirms this at scale." |
| §3.2.2 | "**Three signals include** the channel counter a, …, **which are** therefore each delayed" | "Three signals, namely the channel counter a, …, are therefore each delayed" |
| §3.2.2 | "when all channels of the layer are summed, **result** can be latched" | "…are summed, so the result can be latched" |
| §3.2.4 | "eight main states encoded in 3 bits, **in** Table 3.15" | "…, described in Table 3.15" |
| §3.2.1 | "rescales to **INT8 version** and applies max-pooling" | "rescales to INT8 and applies max-pooling" |
| §3.1.2 | "at a learning rate of 10⁻³ **to** rapid recovery" | "…of 10⁻³, for rapid recovery" |
| §2.3 | "reduces to predicted probability minus label, which **are** cheap" | "…which is cheap" |
| §4.1.1 | "reflecting misses clinically important for pathological classes" | "reflecting misses, which are clinically important…" |
| §4.3.1 | "the scaled **value bias** , the number of shift bits" | "the scaled bias value, the number of shift bits" |

Plus: a stray `)` left after the inline formula in §3.1.3; a missing full stop at the end of §3.2.2; spaces before a comma (§3.2.2, §4.3.1) and before a colon (§3.2.5); four comma splices (§2.1.1, §2.2.2, §2.5, §3.1.2); a missing comma closing a relative clause (§3.1.3); "The static part **moreover** reaches 77%"; and a "two-stage pipeline" in §3.2.3 that contradicted the three cycles described in the same sentence.

One consistency fix: `convolution–pool` (en-dash) appeared 4 times against 17 hyphenated `convolution-pool`. The two body occurrences are now hyphens — see §6 for the two in headings, which I left alone.

---

## 3. Em-dashes — the AI marker that got *added* (16 removed)

The rewrite pass inserted em-dashes into 16 body sentences that previously used ordinary commas or colons:

> "Many arrhythmias **—** atrial fibrillation being the most representative **—** are paroxysmal…"
> "…statistical accuracy **—** which can hide mutually compensating deviations **—** and compares…"
> "Chapter 1 **—** Introduction." … "Chapter 5 **—** Conclusion."
> "The energy per inference **—** decisive for battery-powered wearables **—** is:"

Dense paired em-dashes are the most-cited tell of LLM-generated English. All 16 are back to commas or colons; one em-dash remains in the body, which is fine.

---

## 4. AI-voice rewrites (13)

The residual tells were the antithesis **"X, not Y"** (over a dozen instances) and the **cleft sentence**:

| Was | Now |
|---|---|
| "**What is required is** an automated cardiac monitoring system that operates…" | "This creates a need for automated cardiac monitoring that runs…" |
| "The obstacle is thus **not** a still more accurate model, **but** fitting an already accurate model into an acceptable device form." | "The obstacle is therefore one of deployment rather than of accuracy: accurate models already exist, and the difficulty lies in fitting one into an acceptable device form." |
| "with no bandwidth bottleneck, **what must be optimized is** logic and energy" | "…the quantities to optimize become logic and energy" |
| "…needs one DSP block per multiplication. **This is the whole motivation.**" | "…per multiplication, which is the reason for the choice." |
| "**Several limitations must also be stated in order to situate these conclusions correctly.**" | "These results should be read against four limitations." |
| "**This is a strong advantage** for real-time monitoring, since response time is guaranteed strictly, **not merely** on average." | "For real-time monitoring this matters, because the response time is a hard bound rather than an average." |

Also folded away: "This is the most delicate timing detail of the engine", "Interpretation matters:", "The distinction **is decisive** for hardware", "The study thus **draws a broader conclusion**", "The system is now complete", and one absolute participial clause in §4.4.

---

## 5. Chapter 1 vs Chapter 5

**The objective structure is sound.** §1.1 sets out four groups of tasks — model, quantization, hardware design and verification, board deployment — and §5.1 reports back under exactly those four headings, in the same order. Every number in §5.1 matches Chapter 4.

**But §1.3 contradicted §5.2 on power.**

> **§1.3 (was):** "Third, on deployment, every performance figure reported **is measured** on the synthesized design programmed onto the board."
>
> **§5.2 (unchanged):** "the power and energy figures **are not yet measurements**. The 536.08 mW **was estimated** with PowerPlay … and the tool itself rates the confidence as **low**."

Your abstract is honest about this ("an estimated energy consumption of 27.96 μJ"); only Chapter 1 overclaimed. Fixed in four places — §1.2, §1.3 (twice) and §5.1 — each now separating what is measured (resources, F<sub>max</sub>, latency) from what is estimated (power), with a forward pointer to §5.2.

Two more Chapter 1 corrections:

- §1.2 said the model is "**cross-validated** on Georgia". Georgia is a held-out external set used zero-shot; Chapter 4 correctly calls it cross-dataset testing. Cross-validation is a different procedure.
- §3.1.2 gave the pruning reduction as **48.6%** where §4.2.1 and §5.1 both say **48.55%**.

---

## 6. Cross-references and numbering (20 fixed)

- **§3.1.3 cited Table 3.3** for the quantization parameters. Those are in **Table 3.4**; Table 3.3 is the layer configuration. → corrected.
- **§3.2.2 said "Table 3.10 lists the port interface."** Table 3.10 maps sliding-window cells to multiplication branches; the port interface is Table 3.9. → description corrected.
- **Chapter 4 equation numbering skipped (4.5)** — the sequence ran 4.1, 4.2, 4.3, 4.4, **4.6**, 4.7, 4.8, 4.9, 4.10. Renumbered to 4.1–4.9, and the reference in §4.5.2 updated from "(4.7) and (4.8)" to "(4.6) and (4.7)".
- **Nine tables and three figures had a caption but were never mentioned in the running text:** Tables 2.2, 3.2, 3.3, 3.5, 3.7, 3.9, 3.12, 3.14 and Figures 3.4, 3.5, 3.10, 4.9. A short citing clause was added to the nearest preceding sentence in each case.

---

## 7. Numbers — independently recomputed, all correct

| Claim | Check |
|---|---|
| 1,244 parameters, config (4, 8, 8, 16) | 24 + 168 + 328 + 656 + 68 ✓ |
| 640 parameters, config (4, 4, 8, 8) | 24 + 84 + 168 + 328 + 36 ✓ |
| 48.55% reduction | 604 / 1244 = 48.553% ✓ |
| Conv4 = 328 params | 8 × 8 × 5 + 8 ✓ |
| Lengths 2500 → 500 → 100 → 20 → 4 | consistent with pool S = 5, GAP over 4 values, 512-cell banks ✓ |
| 5,216 cycles | (2500+17)+(2000+31)+(400+31)+(160+51)+22+4 ✓ |
| 52.16 µs at 100 MHz | 5216 / 100e6 ✓ |
| 19,172 inferences/s | 1 / 52.16 µs = 19,172.4 ✓ |
| 27.96 µJ | 536.08 mW × 52.16 µs ✓ |
| 2,148 ALMs = 5%; 28 DSP = 25%; 20 M10K = 4% | / 41,910, / 112, / 553 ✓ |
| 4,688/4,973 = 94.27%; 5,077/5,459 = 93.00% | ✓ |
| Cyclone V 5CSXFC6D6F31C6 | 41,910 ALMs, 553 M10K, 112 DSP — matches the Intel handbook ✓ |

The cycle model in §4.5.2 is the strongest part of the results chapter: the measured overheads reproduce 5 × (input channels) + 11 exactly for Conv2–Conv4, and the extra cycle on Conv1 is correctly explained.

---

## 8b. Applied on request (72 more)

- **Two broken headings fixed.** §3.3.2 "Integration system with the JTAG-to-Avalon IP to communicate with a PC" → "Integration **of the** system with the JTAG-to-Avalon IP **for communication** with a PC". §4.3.1 "convolution**–**pool" → "convolution-pool". The matching TOC lines were left alone deliberately: the TOC is a Word field and regenerates from the headings on Ctrl+A → F9. Editing its text by hand destroys the `<w:tab/>` elements that hold the page-number column.
- **Caption punctuation unified — 38 captions.** Chapter 2 and Chapter 4 captions now end with a full stop like Chapter 3's, in the body and in the List of Figures / List of Tables.
- **28 long tables may now break across pages** (`cantSplit` removed). The 10 short tables keep it, so they will not split awkwardly. Header-row repetition is untouched.

## 8. Left for you

1. **Press Ctrl+A then F9 first.** The TOC is a field and will pick up the two corrected headings, including the stray extra tabs currently on lines 3.3.1 and 3.3.2. Nothing else in the document depends on it.
2. **The List of Figures and List of Tables are typed text, not fields.** F9 will not touch their page numbers — check those by hand once pagination is final, as the last step before printing.
3. **List of Abbreviations** (you chose to leave this): HDL, LSB, LUT and RAM are defined but never used; M10K, SNOMED-CT, MM (Avalon-MM), TP/TN/FP/FN and TPR/FPR are used but not listed; "QRS — QRS complex" defines the term with itself.

---

## 9. Layout pass — indentation and vertical spacing (243 adjustments)

### Why the indents were uneven

The root cause is in the **`Normal` style itself**: it carries `left = 1.20 cm` with a `−1.20 cm` hanging indent. So any paragraph that does not explicitly override it puts its first line at 0 and every wrapped line at 1.20 cm. Most body paragraphs *did* override it — but not with the same numbers, and some not at all. The result was ten different indent positions doing the work of five.

| Class | Values found before | Now |
|---|---|---|
| Body prose | 1.0 cm (113×), **0.99 cm** (57×), 1.09, 1.10, 1.12, 1.27, and some inheriting the style's 1.20 hanging | **first line 1.0 cm, left 0** |
| Bullets, level 1 | left 1.00 / 1.02 / 0.95 / 0.48, some with a −0.5 hanging and some with none | **left 1.0 cm, hanging −0.5** |
| Bullets, level 2 (under a lead-in) | left 1.50 with hanging −0.50 *and* −0.51 | **left 1.5 cm, hanging −0.5** |
| Table / figure captions | mostly inheriting the 1.20 hanging; three at 0.10 or 1.19 | **flush left, 0 / 0** |
| Numbered equation lines | inheriting −1.20 first-line, i.e. starting outside the left margin | **0 / 0** (they are positioned by tabs) |
| List of Figures entries | 22 at −2.01 / 2.01, two at −3.00 / 3.00 | **−2.01 / 2.01** |

The 0.99 vs 1.00 pair is the one that reads as "almost aligned but not quite" — 560 twips against 567. Sixty-three paragraphs were on the wrong side of it.

### Why the bottom of pages was running tall

Three causes, two now fixed:

1. **Stacked blank lines.** The body held 122 empty paragraphs, including six clusters of 2–4 in a row. At 1.5 line spacing each is about 0.8 cm. Those six clusters are now single blank lines — **9 paragraphs removed, about 5.8 cm recovered**. No blank line was removed entirely; every gap still has one.
2. **Captions detaching from their tables.** The whole document had exactly **one** paragraph with "keep with next" set, so a caption could sit alone at the foot of a page while its table jumped to the next. **60 captions** are now bound to what follows.
3. **`cantSplit` on 38 of 46 tables** — when a table cannot fit in the space left on a page, Word moves the whole table down and leaves the gap. I did **not** change this, because letting short tables split across pages usually looks worse. If a specific page still opens up badly, that table is the one to look at.

Text is untouched by this pass: the sequence of (style, text) for all 611 non-empty paragraphs is identical before and after, as are the run, bold and italic counts, all 46 tables, 25 images and 31 equations.

---

## 10. Reviewing the changes

Word → **Review → Compare → Compare**. Original = the file you attached; Revised = `22200034_Thesis_EN_edited.docx`. That gives a redline of all 93 edits to accept or reject individually.
