# from pandas import DataFrame

# from cmf import make_deduper, process, query
# from cmf.clean import company_name, postcode_to_area
# from cmf.dedupers import Naive
# from cmf.helpers import cleaner, cleaners, selector


# def test_naive():
#     # Select
#     select_exp = selector(
#         table="hmrc.trade__exporters", fields=["id", "company_name", "postcode"]
#     )
#     exp_sample = query(select=select_exp, sample=0.05)

#     # Clean
#     cleaner_name = cleaner(function=company_name, arguments={"column": "company_name"})
#     cleaner_pc = cleaner(function=postcode_to_area, arguments={"column": "postcode"})
#     cleaner_name_pc = cleaners(cleaner_name, cleaner_pc)

#     exp_sample_cleaned = process(data=exp_sample, pipeline=cleaner_name_pc)

#     exp_naive_deduper = make_deduper(
#         dedupe_run_name="basic_hmrc_exp",
#         description="""
#             Clean company name, extract postcode area
#         """,
#         deduper=Naive,
#         data_source="hmrc.trade__exporters",
#         data=exp_sample_cleaned,
#         dedupe_settings={"id": "id", "unique_fields": ["company_name", "postcode"]},
#     )

#     exp_deduped = exp_naive_deduper()

#     exp_deduped_df = exp_deduped.to_df()

#     assert isinstance(exp_deduped_df, DataFrame)
#     assert len(exp_deduped_df.index) > 0
