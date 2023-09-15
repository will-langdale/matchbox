# tied_3t_e2

A tied match with three tables and two entities.

Verify it's correct by filtering the probabilities table for "1", then sorting by ID 
ascending to see that leo_inc_t2 and leo_inc_t3 will be given precedence for cluster 1
over will_inc_t2 and will_inc_t3 by the arbitrary mechanic of alphabetical sorting.

will_inc_t1 and leo_inc_t1 remain in the original clusters as they're inserted, not 
matched.