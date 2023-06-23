from src.data import utils as du
from src.features.clean_complex import clean_raw_data

from splink.duckdb.linker import DuckDBLinker
import duckdb


class CompanyMatchingDatasets:
    def __init__(self, sample: int = None):

        self.datasets_and_readfuncs = {
            '"companieshouse"."companies"': self.comp_house_read(sample),
            '"dit"."data_hub__companies"': self.data_hub_read(sample),
            '"hmrc"."trade__exporters"': self.hmrc_exporters_read(sample),
            '"dit"."export_wins__wins_dataset"': self.export_wins_read(sample),
        }
        self.datasets_and_readfuncs_clean = {}
        self.connection = duckdb.connect()
        # self.logger = logging.getLogger(__name__)

        for table in self.datasets_and_readfuncs.keys():
            table_clean = self._clean_table_name(table)
            self.datasets_and_readfuncs_clean[
                table_clean
            ] = self.datasets_and_readfuncs[table]
            self.connection.register(
                f"{table_clean}", self.datasets_and_readfuncs_clean[table_clean]
            )

    def _clean_table_name(self, name):
        return name.replace('"', "").replace(".", "_")

    def _generate_data_log(self, df, df_name, sample):
        msg = f"{df_name}: {len(df)} items loaded"
        if sample is not None:
            msg += ". Sampling is ENABLED"
        return msg

    def linker(self, settings: dict = None):
        return DuckDBLinker(
            list(self.datasets_and_readfuncs_clean.keys()),
            settings_dict=settings,
            connection=self.connection,
        )

    def data_hub_read(self, sample: int = None):
        """
        Read in Data Hub data
        Args:
            sample [int, default: None]: A size of random sample to draw
        Returns: dataset of dit.data_hub__companies
        """

        limit = ""

        if sample is not None:
            limit = f"order by random() limit {sample}"

        dsname = "dit.data_hub__companies"

        cols = """
            id::text as unique_id,
            company_number,
            name as company_name,
            string_to_array(btrim(trading_names, '[]'), ', ') as secondary_names,
            address_postcode as postcode
        """

        query = f"""
            select {cols}
            from {dsname}
            where archived is False
            {limit}
        """

        df_dh = du.query(sql=query)

        # self.logger.info(
        #     self._generate_data_log(
        #         df_dh,
        #         dsname,
        #         sample
        #     )
        # )

        df_dh_clean = clean_raw_data(df_dh)

        # self.logger.info(f"{dsname} cleaned")

        return df_dh_clean

    def comp_house_read(self, sample: int = None):
        """
        Read in Companies House companies data
        Args:
            sample [int, default: None]: A size of random sample to draw
        Returns: dataset of companieshouse.companies
        """

        limit = ""

        if sample is not None:
            limit = f"order by random() limit {sample}"

        dsname = "companieshouse.companies"

        cols = """
            id::text as unique_id,
            company_number,
            company_name,
            array_remove(
                array[
                    previous_name_1,
                    previous_name_2,
                    previous_name_3,
                    previous_name_4,
                    previous_name_5,
                    previous_name_6
                ],
                ''
            ) as secondary_names,
            postcode
        """

        query = f"""
            select {cols}
            from {dsname}
            {limit}
        """

        df_ch = du.query(sql=query)

        # self.logger.info(
        #     self._generate_data_log(
        #         df_ch,
        #         dsname,
        #         sample
        #     )
        # )

        df_ch_clean = clean_raw_data(df_ch)

        # self.logger.info(f"{dsname} cleaned")

        return df_ch_clean

    def hmrc_exporters_read(self, sample: int = None):
        """
        Read in HMRC exporters company data
        Args:
            sample [int, default: None]: A size of random sample to draw
        Returns: dataset of hmrc.trade__exporters
        """

        limit = ""

        if sample is not None:
            limit = f"order by random() limit {sample}"

        dsname = "hmrc.trade__exporters"

        cols = """
            id::text as unique_id,
            null as company_number,
            company_name,
            null as secondary_names,
            postcode
        """

        query = f"""
            select {cols}
            from {dsname}
            {limit}
        """

        df_ex = du.query(sql=query)

        # self.logger.info(
        #     self._generate_data_log(
        #         df_ex,
        #         dsname,
        #         sample
        #     )
        # )

        df_ex_clean = clean_raw_data(df_ex)

        # self.logger.info(f"{dsname} cleaned")

        return df_ex_clean

    def export_wins_read(self, sample: int = None):
        """
        Read in Export Wins companies data
        Args:
            sample [int, default: None]: A size of random sample to draw
        Returns: dataset of dit.export_wins__wins_dataset
        """

        limit = ""

        if sample is not None:
            limit = f"order by random() limit {sample}"

        dsname = "dit.export_wins__wins_dataset"

        cols = """
            id::text as unique_id,
            cdms_reference as company_number,
            company_name,
            null as secondary_names,
            null as postcode
        """

        query = f"""
            select {cols}
            from {dsname}
            {limit}
        """

        df_ew = du.query(sql=query)

        # self.logger.info(
        #     self._generate_data_log(
        #         df_ex,
        #         dsname,
        #         sample
        #     )
        # )

        df_ew_clean = clean_raw_data(df_ew)

        # self.logger.info(f"{dsname} cleaned")

        return df_ew_clean
