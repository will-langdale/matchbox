def cms_original_clean_company_name_general(input_column):
    """
    Replicates the original Company Matching Service company name cleaning
    regex exactly. Intended to help replicate the methodology for comparison.

    The _general_name_simplification version from
    app/algorithm/sql_statements.py#L24.

    Use with any dataset except Companies House.
    """
    regex_1 = (
        r'^the\s|\s?:\s?|\[|\]|\(|\)|\'\'|\*.*\*|&|,|;|"|ltd\.?$|'
        r"limited\.?$|\sllp\.?$|\splc\.?$|\sllc\.?$|\sand\s|\sco[\.|\s]|"
        r"\scompany[\s|$]|\s+"
    )
    regex_2 = (
        r"\s/.*|\s-(\s)?.*|(\s|\.|\(|\_)?duplic.*|\sdupilcat.*|"
        r"\sdupicat.*|\sdissolved.*|\*?do not .*|((\s|\*)?in)?\sliquidat.*|"
        r"\sceased trading.*|\strading as.*|t/a.*|\sacquired by.*|"
        r"\splease\s.*|(do not)?(use)?(.)?(\d{{5}}(\d*)).*|-|\."
    )
    return rf"""
        lower(
            coalesce(
                nullif(
                    regexp_replace(
                        regexp_replace(
                            regexp_replace(
                                {input_column},
                                '{regex_1}',
                                ' ',
                                'gi'
                            ),
                            '{regex_2}',
                            '',
                            'gi'
                        ),
                        '\.|\s',
                        '',
                        'gi'
                    ),
                    ''
                ),
                {input_column}
            )
        )
    """


def cms_original_clean_company_name_ch(input_column):
    """
    Replicates the original Company Matching Service company name cleaning
    regex exactly. Intended to help replicate the methodology for comparison.

    The _ch_name_simplification version from app/algorithm/sql_statements.py#L14.

    Use with Companies House only.
    """
    regex_1 = (
        r'^the\s|\s?:\s?|\[|\]|\(|\)|\'\'|\*.*\*|&|,|;|"|ltd\.?$|limited\.?$|'
        r"\sllp\.?$|\splc\.?$|\sllc\.?$|\sand\s|\sco[\.|\s]|\scompany[\s|$]"
    )
    regex_2 = r"\.|\s"
    return rf"""
        lower(
            coalesce(
                nullif(
                    regexp_replace(
                        regexp_replace(
                            {input_column},
                            '{regex_1}',
                            ' ',
                            'gi'
                        ),
                        '{regex_2}',
                        '',
                        'gi'
                    ),
                    ''
                ),
                {input_column}
            )
        )
    """
