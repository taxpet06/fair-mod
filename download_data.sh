#!/usr/bin/env bash
# Civil Comments, CC0. ~240 MB. The WILDS mirror's raw/ folder carries the annotator fractions used as soft labels.
set -e
cd "$(dirname "$0")"
hf download pietrolesci/civilcomments-wilds --repo-type dataset --local-dir data/wilds
