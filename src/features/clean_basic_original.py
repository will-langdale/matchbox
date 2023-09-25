def cms_original_clean_company_name_general(input_column):
    """
    Replicates the original Company Matching Service company name cleaning
    regex exactly. Intended to help replicate the methodology for comparison.

    The _general_name_simplification version from
    app/algorithm/sql_statements.py#L24.

    Use with any dataset except Companies House.
    """
    regex_1 = (
        r"^the\s|\s?:\s?|\[|\]|\(|\)|''|\*.*\*|&|,|;|\"|ltd\.?$|"
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
        r"^the\s|\s?:\s?|\[|\]|\(|\)|''|\*.*\*|&|,|;|\"|ltd\.?$|limited\.?$|"
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


def cms_original_clean_postcode(input_column):
    """
    Replicates the original Company Matching Service postcode cleaning SQL
    exactly. Intended to help replicate the methodology for comparison.
    """
    return f"lower(replace({input_column}, ' ', ''))"


def cms_original_clean_email(input_column):
    """
    Replicates the original Company Matching Service email cleaning SQL
    exactly. Intended to help replicate the methodology for comparison.
    """
    return f"lower(split_part({input_column}, '@', 2))"


def cms_original_clean_ch_id(input_column):
    """
    Replicates the original Company Matching Service Companies House ID
    cleaning SQL exactly. Intended to help replicate the methodology for
    comparison.
    """
    return f"""
        case when
            lower({input_column}) = ANY(
                '{{notregis, not reg,n/a, none, 0, ""}}'::text[]
            )
        then
            null
        else
            lower({input_column})
        end
    """


def cms_original_clean_cdms_id(input_column):
    """
    Replicates the original Company Matching Service CDMS ID cleaning SQL
    exactly. Intended to help replicate the methodology for comparison.
    """
    return f"regexp_replace({input_column}, '\\D', '', 'g')"
