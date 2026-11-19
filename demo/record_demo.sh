#!/usr/bin/env bash
# Scripted walkthrough of the running app (python app.py), recorded with agent-browser -> fairmod_demo.webm / .mp4
set -e
cd "$(dirname "$0")"
AB=agent-browser
ref() { $AB snapshot -i -c | grep -E "$1" | head -1 | grep -oE "ref=e[0-9]+" | cut -d= -f2; }
pick() { $AB click @$(ref 'combobox "Pick a comment') >/dev/null; $AB wait 500 >/dev/null; $AB click @$(ref "option \".*$1") >/dev/null; $AB wait 900 >/dev/null; }
model() { $AB click @$(ref 'combobox "Classifier"') >/dev/null; $AB wait 400 >/dev/null; $AB click @$(ref "option \"$1\"") >/dev/null; $AB wait 600 >/dev/null; }
score() { $AB click @$(ref 'button "Score comment"') >/dev/null; $AB wait 1500 >/dev/null; $AB wait --text "Calibrated" --timeout 60000 >/dev/null; $AB wait 1200 >/dev/null; }
show() { $AB scroll down "$1" >/dev/null; $AB wait "${2:-2500}" >/dev/null; }
top() { $AB scroll up 3000 >/dev/null; $AB wait 600 >/dev/null; }

$AB open http://127.0.0.1:7860 >/dev/null; $AB set viewport 1400 900 >/dev/null; $AB wait 2000 >/dev/null
$AB record start "$PWD/fairmod_demo.webm" --cursor --fps 12 >/dev/null; $AB wait 2500 >/dev/null

# 1. benign identity statement: TF-IDF ERM flags it, the swap table shows the identity term drives the score
pick "proud gay man"; model "tfidf_erm"; score; show 450 3500; show 500 4000; top
# 2. same comment, best model (DistilBERT + soft labels)
model "bert_distil_soft"; score; show 450 3500; show 500 4000; top
# 3. real test comment defending Muslims, 0/4 raters said toxic, DistilBERT ERM flags it
pick "radical Islamic"; model "bert_distil_erm"; score; show 350 3000; top
model "bert_distil_soft"; score; show 350 3500; show 500 3500; top
# 4. raters split 50/50 (5 of 10): the soft-label model flags it (0.54 vs threshold 0.26) and the app shows the rater split; band 0 gives the same hard decision
pick "female friendly"; score; show 300 3500; top
$AB fill @$(ref 'spinbutton "number input for Defer band') 0 >/dev/null; $AB press Enter >/dev/null; $AB wait 500 >/dev/null; score; show 300 3000; top
$AB fill @$(ref 'spinbutton "number input for Defer band') 0.15 >/dev/null; $AB press Enter >/dev/null
# 5. failure case: toxic comment the model misses
pick "missed toxic"; score; show 300 4000; top
# 6. implicit attack typed by the moderator
pick "go back to where you came from"; score; show 300 3500; top
# 7. group metrics and limitations tabs
$AB click @$(ref 'tab "Group metrics"') >/dev/null; $AB wait 2500 >/dev/null; show 500 3000; show 600 3500; show 700 3500; top
$AB click @$(ref 'tab "Limitations"') >/dev/null; $AB wait 5000 >/dev/null
$AB record stop >/dev/null
ffmpeg -y -loglevel error -i fairmod_demo.webm -c:v libx264 -pix_fmt yuv420p -crf 26 -movflags +faststart fairmod_demo.mp4
echo "done: $(ffprobe -v error -show_entries format=duration -of csv=p=0 fairmod_demo.mp4) s"
