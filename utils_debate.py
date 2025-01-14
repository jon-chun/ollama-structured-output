import pandas as pd
import logging



def read_all_vignettes(filepath: str) -> pd.DataFrame:
    """
    Reads the CSV containing all vignettes from `filepath`.
    Adjust to match your CSV schema (columns, delimiter, etc.).
    """
    logger.info(f"Reading all vignettes from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} total vignettes")
    return df


def select_vignettes(df_all: pd.DataFrame, 
                     strategy: str = "random", 
                     row_ct: int = 30, 
                     arg_extra: Optional[int] = None) -> pd.DataFrame:
    """
    Returns a subset of df_all based on `strategy`, up to `row_ct` rows.
    - strategy='random' => randomly sample row_ct rows with seed=arg_extra
    - strategy='first-nrows' => take first row_ct rows, ignoring arg_extra
    """
    if strategy == "random":
        seed = arg_extra if arg_extra is not None else 42
        logger.info(f"Selecting random {row_ct} rows with seed={seed}")
        df_subset = df_all.sample(n=row_ct, random_state=seed)
    elif strategy == "first-nrows":
        logger.info(f"Selecting first {row_ct} rows from the DataFrame.")
        df_subset = df_all.head(row_ct)
    else:
        logger.warning(f"Unknown strategy={strategy}, returning the entire df_all.")
        df_subset = df_all
    
    logger.info(f"Selected {len(df_subset)} vignettes (strategy={strategy})")
    return df_subset
