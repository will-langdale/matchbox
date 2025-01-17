---
layout: sub-navigation
title: Link and deduplicate
description: Perform linking and deduplication of your data
sectionKey: Client
eleventyNavigation:
  parent: Client
  order: 1
---

You have a dataset you want to link to your organisation's broader network of data.

Your high level process will be:

1. Use `matchbox.query()` to retrieve source data from the perspective of a particular resolution point
2. Use `matchbox.process()` to clean the data with standardised processes
3. Use `matchbox.make_model()` with `matchbox.dedupers` and `matchbox.linkers` to create a new model
4. Generate probabilistic model outputs using `model.run()`
5. Upload the probabilites to matchbox with `results.to_matchbox()`
6. Label data, or use existing data, to decide the probability threshold that you're willing to consider "truth" for your new model
7. Use `model.roc_curve()` and other tools to make your decision
8. Update `model.truth` to codify it

Full documentation to follow.