import pandas as pd

cols = ['A', 'B', 'C', 'D']
import numpy as np
random_df = pd.DataFrame(abs(np.random.randn(5, 4)), columns=cols)

# Get arguments 
dataframe = random_df
color_map = "Blues"
all_columns_same_measure = False

def make_pretty(styler):
    cell_hover = {  # for row hover use <tr> instead of <td>
        'selector': 'td:hover',
        'props': 'background-color: #ffffb3; color: #2b2b2b;'
    }
    headers = {
        'selector': 'th:not(.index_name)',
        'props': 'text-align: center;'
    }
    styler.set_table_styles([cell_hover, headers])
    styler.format(thousands=",")
    styler.background_gradient(cmap=color_map)
    if all_columns_same_measure:
        styler.background_gradient(axis=None)
    styler.set_properties(**{'padding': '0.6em'})
    styler.set_caption("Here's a caption")
    return styler
dataframe.style.pipe(make_pretty)
