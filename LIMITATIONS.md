## What this tool is for

Fair-Mod helps a moderator decide whether to trust a toxicity flag. It shows the score, why the model leaned that
way, and whether the score moves when only an identity term changes. It is **not** meant to remove comments on its own.
The defer band sends uncertain comments to a person.

## Known limitations

- **Labels are noisy crowd judgments.** The median test comment was rated by 4 people (74% by fewer than 10), and
  18% of the test split sits between 30% and 70% toxic. On those comments "correct" means "agrees with most raters".
- **Eight coarse identity groups.** The audit covers male, female, LGBTQ, christian, muslim, other religions, black,
  and white. A clean result on these groups says nothing about groups the data doesn't label, or about intersections.
- **A mention is not the author's identity.** Identity flags record that a comment talks about a group. They don't
  record who wrote it, so this audit cannot measure harm to in-group speakers directly.
- **Dated text.** The comments come from North American news sites, 2015 to 2017.
- **Small models, few seeds.** The strongest model is DistilBERT (66M parameters, one epoch, three seeds). bert-mini
  (11M, 80K-comment subsample, laptop CPU) and TF-IDF have one seed each, so their confidence intervals cover
  test-set sampling, not training randomness.
- **Swap lexicon is crude.** Words like *black*, *white*, *straight*, and *man* are swapped even when they don't refer
  to people ("black hole"). That adds noise to augmentation and to the swap table.
- **Attributions are approximate.** Integrated gradients explains the toxicity logit relative to a [PAD] baseline.
  TF-IDF highlights show unigram weights only and leave out bigrams.
- **Calibration is fit on the WILDS validation split.** It may not hold for comments from other sites or other times.
