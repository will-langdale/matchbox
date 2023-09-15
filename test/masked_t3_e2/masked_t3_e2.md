# masked_t3_e2

A masked match with three tables and two entities.

An explanation of the masking behaviour:

will_inc_t2 should belong to cluster 1, but its probability for 1 (0.7) is lower than 
leo_inc_t2's (0.9).

However, leo_inc_t2's probability for cluster 2 (1) is higher than its probability for 
1 (0.9, as above). Once leo_inc_t2 is assigned to cluster 2, will_inc_t2 can be assigned 
to cluster 1.

The situation is produced again, this time with the clusters reversed, to show order 
doesn't matter. 

leo_inc_t3 should belong to cluster 2, but its probability for 2 (0.7) is lower than 
will_inc_t3's (0.9).

However, will_inc_t3's probability for cluster 1 (1) is higher than its probability for 
2 (0.9, as above). Once will_inc_t3 is assigned to cluster 1, leo_inc_t3 can be assigned 
to cluster 2.

There is no way to easily verify this is the case.
