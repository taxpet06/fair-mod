# Fair-Mod speaker notes (3 minutes)

## 1. cover

Toxicity classifiers learn that identity words predict toxicity, because those words show up so often in abusive training comments. The result is that people talking about their own identity get flagged. Here is my own TF-IDF baseline from the proposal: “I am a gay man” scores 0.86 and the straight version 0.04. My question had two parts: how much of this can mitigation remove without losing accuracy, and can a moderator tool make the bias visible instead of hiding it.

## 2. built

Data: CivilComments-WILDS, about 445 thousand comments, each flagged for which of eight identity groups it mentions. I removed anything in train or validation that also appears in test, and I took soft labels straight from the raw file: the fraction of raters who said toxic. Three model families, so I can tell whether a mitigation works in general or only for one model: TF-IDF, a tiny bert-mini I could train on my laptop, and DistilBERT on the Dartmouth GPU cluster with three seeds. Four mitigations from the proposal, applied to every family. The decision threshold and calibration come from validation only, so the test set never picks anything.

## 3. results

The biggest effect is the model, not the mitigation. DistilBERT ERM beats every TF-IDF variant on AUC and on bias AUC. Among mitigations, soft labels are the only one that never lowers AUC: best AUC and best bias AUC across all three seeds, accuracy within 0.01 of ERM, and the highest worst-group accuracy, though that last gain is within seed noise. But it does not shrink the false-positive gap. Reweighting and group DRO shrink that gap about three times, but they pay for it: worst-group accuracy falls because toxic comments about those same groups get missed, and calibration breaks. So which mitigation is best depends on which error you care about. That is a policy choice, not a modelling one.

## 4. errors

Four things from the error analysis. First, errors concentrate where raters disagreed: 92 percent of missed toxic comments sat between 30 and 70 percent toxic. Second, the template probe: DistilBERT still flags about one in five neutral sentences like “I’m proud to be gay”. Third, augmentation works only on the words it swapped; on held-out identity terms there is no gain. Fourth, integrated gradients: an identity word is still among the top three words in more than half of the confident identity false positives, and more so for the soft-label model. Soft labels probably copy the raters’ own partial sensitivity to identity mentions. That last point is a hypothesis. Finally, I hand-coded 200 errors: the identity mention is the real cause in only 28 of 100 false positives, and the missed toxic comments are mostly implicit hostility or sarcasm, which have no tell-tale words.

## 5. takeaways

Three takeaways. One: model capacity and data did more for identity bias than any of the mitigations. Two: soft labels are the one mitigation that never lowers AUC, and closing the false-positive gap always cost missed attacks on the same groups, so the choice between those errors belongs to people, not the model. Three: that is why the tool defers instead of auto-removing. Now the demo: a benign identity comment the baseline flags, the identity-swap table, a real comment defending Muslims that the ERM model flags, a comment raters split 50-50 on, where the app shows that split next to the model's flag, and a failure case, “go back to where you came from”, which the best model lets through at a toxicity of 0.02.
