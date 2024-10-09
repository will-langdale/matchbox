from pandas import Series, concat

from matchbox.data import utils as du


def test_sha1_conversion(all_companies):
    """Tests SHA1 conversion works as expected."""
    sha1_series_1 = du.columns_to_value_ordered_sha1(
        data=all_companies,
        columns=["id", "company_name", "address", "crn", "duns", "cdms"],
    )

    assert isinstance(sha1_series_1, Series)
    assert len(sha1_series_1) == all_companies.shape[0]

    all_companies_reordered_top = (
        all_companies.head(500)
        .rename(
            columns={
                "company_name": "address",
                "address": "company_name",
                "duns": "crn",
                "crn": "duns",
            }
        )
        .filter(["id", "company_name", "address", "crn", "duns", "cdms"])
    )

    all_companies_reodered = concat(
        [all_companies_reordered_top, all_companies.tail(500)]
    )

    sha1_series_2 = du.columns_to_value_ordered_sha1(
        data=all_companies_reodered,
        columns=["id", "company_name", "address", "crn", "duns", "cdms"],
    )

    assert sha1_series_1.equals(sha1_series_2)
