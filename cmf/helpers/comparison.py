from typing import Dict, List


def comparison(sql_condition: str) -> Dict[str, str]:
    return {"comparison": sql_condition}


def comparisons(*comparison: Dict[str, str]) -> Dict[str, List]:
    return {"comparisons": list(comparison)}


if __name__ == "__main__":
    pass
