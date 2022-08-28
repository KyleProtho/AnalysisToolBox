# Load packages
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Declare function
def CreateAssociationRules(dataframe,
                           key_column,
                           items_column,
                           support_threshold = .20,
                           confidence_threshold = .50):
    # Group and summarize (as list) items to key column
    dataframe = dataframe.groupby(key_column)[items_column].apply(list)
    dataframe = dataframe.reset_index()

    # Create association rule mining-ready dataset of transactions
    transactions = TransactionEncoder()
    df_transactions = transactions.fit(dataframe[items_column]).transform(dataframe[items_column])
    df_transactions = pd.DataFrame(df_transactions,
                                columns=transactions.columns_)

    # Create apriori model
    if support_threshold != None:
        df_association_rules = apriori(
            df_transactions, 
            min_support = support_threshold,
            use_colnames = True, 
            verbose = 1
        )
    else:
        df_association_rules = apriori(
            df_transactions,
            use_colnames = True, 
            verbose = 1
        )

    # Create association rules based on apriori model
    df_association_rules = association_rules(
        df_association_rules, 
        metric = "confidence", 
        min_threshold = confidence_threshold
    )
    df_association_rules = df_association_rules.sort_values(
        by=['confidence', 'consequents'],
        ascending=[False, True]
    )

    # Return association rules
    return(df_association_rules)
