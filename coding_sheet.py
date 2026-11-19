"""Hand-coding workbook for the 200 errors in results/errors_to_code.csv.

python coding_sheet.py make   -> results/errors_to_code.xlsx (refuses to overwrite: it holds your coding)
python coding_sheet.py read [FILE.xlsx]  -> copies manual_category back into results/errors_to_code.csv, prints FP/FN x category

The author's finished labels are in results/errors_hand_coded.csv (id, error, human_label, human_note);
report/build_tables.py reads that file.
"""
import math
import sys

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

RESULTS = __import__("pathlib").Path(__file__).parent / "results"

CSV, XLSX = RESULTS / "errors_to_code.csv", RESULTS / "errors_to_code.xlsx"
N = 200

# (category, when to use it, example). Order = dropdown order.
CATEGORIES = [
    ("Identity mention, not hostile", "Names a group (gay, Muslim, Black, women...) but is not attacking it. Typical false alarm.",
     "\"As a gay man, I support this bill.\""),
    ("Quoting or reporting toxicity", "Repeats, quotes, or describes someone else's offensive words without endorsing them.",
     "\"He called her a 'stupid cow' on air, which is unacceptable.\""),
    ("Harsh but civil disagreement", "Strong political or personal criticism, but no insults, slurs, or threats.",
     "\"This mayor is incompetent and his budget is a disaster.\""),
    ("Rude words, not aimed at anyone", "Swearing or crude words used for emphasis, not to attack a person or group.",
     "\"Damn, this winter has been brutal.\""),
    ("Implicit or coded hostility", "Hostile or demeaning without obvious bad words: stereotypes, dog whistles, dehumanizing.",
     "\"People like them just don't belong in this country.\""),
    ("Sarcasm or irony", "The literal words mean something different from the intended meaning.",
     "\"Oh sure, because you're SUCH a genius.\""),
    ("Gold label looks wrong", "You confidently disagree with the human raters: the model was actually right.",
     "A plainly polite comment labeled toxic, or a clear insult labeled fine."),
    ("Genuinely ambiguous", "Reasonable people could go either way; you can't call it.",
     "\"Typical. Can't expect anything else from that crowd.\""),
    ("Other (explain in Notes)", "None of the above fits. Write a few words in the Notes column.", ""),
]

ARIAL = "Arial"
YELLOW = PatternFill("solid", start_color="FFF2CC")
HEAD = PatternFill("solid", start_color="1F4E78")
GREY = PatternFill("solid", start_color="F2F2F2")
THIN = Border(*(Side(style="thin", color="BFBFBF"),) * 4)


def font(**kw):
    return Font(name=ARIAL, **kw)


def header(ws, row, values):
    for col, v in enumerate(values, 1):
        c = ws.cell(row, col, v)
        c.font, c.fill = font(bold=True, color="FFFFFF"), HEAD
        c.alignment = Alignment(wrap_text=True, vertical="center")


def make():
    if XLSX.exists():
        sys.exit(f"{XLSX} already exists; delete it first if you really want a blank one.")
    d = pd.read_csv(CSV)
    assert len(d) == N, len(d)
    wb = Workbook()

    # ---------- 1. Instructions ----------
    ws = wb.active
    ws.title = "Instructions"
    ws.column_dimensions["A"].width = 4
    ws.column_dimensions["B"].width = 110
    lines = [
        ("How to hand-code the 200 model errors", "title"),
        ("", None),
        ("WHAT YOU'RE DOING", "h"),
        ("The best model (DistilBERT + soft labels) got these 200 test comments wrong: 100 false positives and 100 false "
         "negatives, picked at random (seed 0). For each one, pick the single reason that best explains the mistake.", None),
        ("", None),
        ("STEPS", "h"),
        ("1.  Open the \"Coding\" tab.", None),
        ("2.  Read the comment in column G.", None),
        ("3.  Click the yellow cell in column H (\"Your category\") and pick a reason from the dropdown. "
         "Only dropdown values are accepted, so you can't make a typo.", None),
        ("4.  Optional: write a short note in column I (required only if you picked \"Other\").", None),
        ("5.  Google Sheets saves automatically. You can stop and come back any time; the progress counter below updates itself.", None),
        ("6.  When all 200 are done: File > Download > Microsoft Excel (.xlsx). Then, in the fairmod folder, run:", None),
        ("     python coding_sheet.py read ~/Downloads/errors_to_code.xlsx", None),
        ("     That copies your answers into results/errors_to_code.csv and prints the FP/FN x category table for the report.", None),
        ("", None),
        ("ONLY EDIT THE YELLOW CELLS (columns H and I on the Coding tab). Everything else is data.", "warn"),
        ("", None),
        ("WHAT THE COLUMNS MEAN", "h"),
        ("Error = FP (false positive): the model FLAGGED the comment as toxic, but most human raters said it is NOT toxic.", None),
        ("Error = FN (false negative): the model LET IT THROUGH, but most human raters said it IS toxic.", None),
        ("Model score: the model's toxicity probability (0-1). It flags anything at or above about 0.26.", None),
        ("Raters' toxic share: fraction of human raters who called it toxic. 0.5 or more = gold label \"toxic\".", None),
        ("# raters: how many people rated it. Few raters (under 6) = a noisier gold label.", None),
        ("", None),
        ("TIPS", "h"),
        ("- Pick the reason for the ERROR, not a description of the comment. Ask: \"why did the model get this wrong?\"", None),
        ("- If two categories fit, choose the one that most directly explains the mistake.", None),
        ("- \"Gold label looks wrong\" means you think the model was right and the raters were wrong. Use it when confident.", None),
        ("- Don't overthink: about 15-30 seconds per comment. Use \"Genuinely ambiguous\" when you truly can't decide.", None),
        ("- The full list with definitions and examples is on the \"Categories\" tab. A summary table is on the \"Summary\" tab.", None),
        ("", None),
        ("EXAMPLE OF A FINISHED ROW", "h"),
        ("Comment: \"Muslims in my town organized a food drive for the flood victims.\"   Error: FP   Model score: 0.62", None),
        ("-> Your category: Identity mention, not hostile      Notes: (blank)", None),
        ("", None),
        ("PROGRESS", "h"),
    ]
    for i, (text, kind) in enumerate(lines, 1):
        c = ws.cell(i, 2, text)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        c.font = {"title": font(bold=True, size=16), "h": font(bold=True, size=12, color="1F4E78"),
                  "warn": font(bold=True, color="C00000")}.get(kind, font())
        if kind == "warn":
            c.fill = YELLOW
    r = len(lines) + 1
    ws.cell(r, 2, f'="Coded so far: "&COUNTA(Coding!H2:H{N + 1})&" of {N}  ("&TEXT(COUNTA(Coding!H2:H{N + 1})/{N},"0%")&")"').font = font(bold=True, size=12)

    # ---------- 2. Categories ----------
    cat = wb.create_sheet("Categories")
    header(cat, 1, ["Category (dropdown value)", "When to use it", "Example"])
    for i, row in enumerate(CATEGORIES, 2):
        for j, v in enumerate(row, 1):
            c = cat.cell(i, j, v)
            c.font, c.border = font(bold=j == 1), THIN
            c.alignment = Alignment(wrap_text=True, vertical="top")
    for col, w in zip("ABC", (32, 70, 55)):
        cat.column_dimensions[col].width = w
    cat.freeze_panes = "A2"
    last_cat = len(CATEGORIES) + 1

    # ---------- 3. Coding ----------
    cs = wb.create_sheet("Coding", 1)
    cols = ["#", "id", "Error", "Model score", "Raters' toxic share", "# raters", "Comment",
            "Your category  (pick from dropdown)", "Notes (optional)"]
    header(cs, 1, cols)
    cs.row_dimensions[1].height = 32
    for i, x in enumerate(d.itertuples(), 2):
        vals = [i - 1, x.id, x.error, round(x.p_bert_distil_soft, 3), round(x.toxicity_soft, 2),
                x.toxicity_annotator_count, x.text]
        for j, v in enumerate(vals, 1):
            c = cs.cell(i, j, v)
            c.font, c.border, c.fill = font(), THIN, GREY
            c.alignment = Alignment(wrap_text=j == 7, vertical="top", horizontal="left" if j == 7 else "center")
        for j in (8, 9):
            c = cs.cell(i, j)
            c.font, c.border, c.fill = font(), THIN, YELLOW
            c.alignment = Alignment(wrap_text=True, vertical="top")
        # ponytail: row height estimated from length (~105 chars/line); Excel won't autofit openpyxl rows
        cs.row_dimensions[i].height = max(30, 15 * math.ceil(len(x.text) / 105))
    for col, w in zip("ABCDEFGHI", (5, 10, 7, 9, 10, 8, 95, 32, 30)):
        cs.column_dimensions[col].width = w
    cs.freeze_panes = "H2"  # header + comment stay visible while scrolling
    cs.auto_filter.ref = f"A1:I{N + 1}"
    dv = DataValidation(type="list", formula1=f"Categories!$A$2:$A${last_cat}", allow_blank=True,
                        showErrorMessage=True, errorTitle="Not a category",
                        error="Please pick a category from the dropdown list.",
                        showInputMessage=True, promptTitle="Your category", prompt="Pick the main reason the model got this wrong.")
    cs.add_data_validation(dv)
    dv.add(f"H2:H{N + 1}")

    # ---------- 4. Summary ----------
    sm = wb.create_sheet("Summary")
    header(sm, 1, ["Category", "FP", "FN", "Total", "% of coded"])
    rng = f"Coding!$H$2:$H${N + 1}"
    err = f"Coding!$C$2:$C${N + 1}"
    for i in range(2, last_cat + 1):
        sm.cell(i, 1, f"=Categories!A{i}")
        sm.cell(i, 2, f'=COUNTIFS({rng},$A{i},{err},"FP")')
        sm.cell(i, 3, f'=COUNTIFS({rng},$A{i},{err},"FN")')
        sm.cell(i, 4, f"=B{i}+C{i}")
        sm.cell(i, 5, f"=IF($D${last_cat + 2}=0,0,D{i}/$D${last_cat + 2})").number_format = "0%"
    t = last_cat + 1
    sm.cell(t, 1, "Not coded yet")
    sm.cell(t, 2, f'=COUNTIFS({err},"FP")-SUM(B2:B{last_cat})')
    sm.cell(t, 3, f'=COUNTIFS({err},"FN")-SUM(C2:C{last_cat})')
    sm.cell(t, 4, f"=B{t}+C{t}")
    sm.cell(t + 1, 1, "Total coded")
    for j, col in ((2, "B"), (3, "C"), (4, "D")):
        sm.cell(t + 1, j, f"=SUM({col}2:{col}{last_cat})")
    for row in sm.iter_rows(min_row=2, max_row=t + 1):
        for c in row:
            c.font, c.border = font(bold=c.row > last_cat), THIN
    sm.column_dimensions["A"].width = 32
    sm.cell(t + 3, 1, "Fills in automatically from the Coding tab. Don't edit.").font = font(italic=True, color="808080")

    wb.calculation.fullCalcOnLoad = True  # Excel/LibreOffice compute the progress + summary formulas on open
    wb.save(XLSX)
    print(f"wrote {XLSX}")


def read(path=XLSX):
    x = pd.read_excel(path, sheet_name="Coding")
    x = x.rename(columns={x.columns[7]: "manual_category", x.columns[8]: "manual_notes"})
    valid = {c for c, _, _ in CATEGORIES}
    missing = x.manual_category.isna().sum()
    bad = set(x.manual_category.dropna()) - valid
    assert not bad, f"unknown categories: {bad}"
    other_no_note = x[(x.manual_category == "Other (explain in Notes)") & x.manual_notes.isna()]
    assert other_no_note.empty, f"'Other' without a note in rows {other_no_note['#'].tolist()}"
    d = pd.read_csv(CSV).drop(columns=["manual_category", "manual_notes"], errors="ignore")
    d = d.merge(x[["id", "manual_category", "manual_notes"]], on="id", how="left", validate="1:1")
    assert len(d) == N
    d.to_csv(CSV, index=False)
    print(f"updated {CSV}; {N - missing}/{N} coded" + (f" ({missing} still blank)" if missing else ""))
    print(pd.crosstab(d.manual_category, d.error, margins=True))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["make"]
    {"make": make, "read": read}[cmd](*args)
