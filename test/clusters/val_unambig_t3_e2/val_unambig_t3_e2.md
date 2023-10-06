# val_unambig_t3_e2

A unambiguous match with three tables and two entities, contradicted by an entry in the 
validation table.

Verify it's correct by filtering the probabilities table for "1", then checking that 
the validation table overrides this output by swapping the *_t3 clusters.

Note the probabilities table is identical to unambig_t3_e2. Only the validate table 
changes.
