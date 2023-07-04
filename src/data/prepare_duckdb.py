# import click
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path

import src.data.utils as du


def prepare_local_duckdb(input_dir, duckdb_dir=du.DEFAULT_DUCKDB_PATH, overwrite=True):
    con = du.get_duckdb_connection(path=duckdb_dir.as_posix())

    if overwrite:
        del con
        Path(duckdb_dir).unlink()
        con = du.get_duckdb_connection(path=duckdb_dir.as_posix())

    dfs_to_load = du.build_alias_path_dict(input_dir)

    # Create lookups with autoincrementing primary key

    con.query(
        """
        create sequence unique_id_lookup_pk start 1;
        create sequence table_alias_lookup_pk start 1;
        create table unique_id_lookup(
            id integer primary key default nextval('unique_id_lookup_pk'),
            unique_id varchar
        );
        create table table_alias_lookup(
            id integer primary key default nextval('table_alias_lookup_pk'),
            unique_id varchar
        );
    """
    )

    # Insert data into lookups from files

    select_list = [f"select unique_id from '{df}'" for df in list(dfs_to_load.values())]

    sql = " union ".join(select_list)

    con.query(
        f"""
        insert into unique_id_lookup
        by name ({sql});
        insert into table_alias_lookup
        values ({list(dfs_to_load.keys())});
    """
    )

    # Load cleaned datasets into database, replacing lookup values

    for df in dfs_to_load.keys():
        con.query(
            f"""
            create table {df} as
            select * from {dfs_to_load[df]} d
                join unique_id_lookup l on
                    (d.unique_id = l.unique_key);
            alter table {df} drop unique_id;
            alter table {df} rename column id to unique_id;
        """
        )


def main():
    """
    Entrypoint
    """

    prepare_local_duckdb(
        input_dir="company-matching__06-26-23_11-40-51", overwrite=True
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=du.LOG_FMT)

    load_dotenv(find_dotenv())

    main()
