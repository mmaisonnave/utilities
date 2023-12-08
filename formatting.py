

def wrap_tabular(latex_table:str)->str:
    """
    This function ...
    """
    table_begin =  r'\begin{table}[!ht]'
    resizebox = r'\centering\resizebox{\textwidth}{!}{'
    caption = r'}\caption{Caption}\label{tab:my_label}'
    end_table = r'\end{table}'

    return table_begin + '\n' + resizebox + '\n' + latex_table + caption + '\n' + end_table