import numpy as np
import pandas as pd
import seaborn as sns


def arr_to_df(coefs_noisy, n_arr, coefs_id):
    """Convert numpy array of noisy coefs to dataframe for plotting."""
    out = pd.DataFrame(coefs_noisy, columns=n_arr)
    out = pd.DataFrame(out.stack()).reset_index()
    out.columns = ['component', 'n', 'value']
    out = out.assign(id=coefs_id)

    return out


def vec_to_df(coefs, n_arr, coefs_id):
    """Convert 1-d numpy array to dataframe for plotting."""
    out = pd.DataFrame({'component': 'L2_dist', 'n': n_arr, 'value': coefs,
                        'id': coefs_id})
    return out


def transform_metrics(res, reference_col, scale=0.5):
    """
    Center and scale metric values using their respective reference values.

    - For 'risk' and 'risk_change', uses reference_col.
    - For 'gap_FPR', uses `epsilon_pos`.
    - For 'gap_FNR', uses `epsilon_neg`.

    Args:
        res (pd.DataFrame): Input long-format DataFrame with columns including
                            'metric', 'value', reference_col, 'epsilon_pos', 'epsilon_neg'.
        scale (float): Exponent used in scaling (e.g., 0.5 for sqrt(n)).
        id_vars (tuple): Columns to group by (usually 'mc_iter' and 'n').

    Returns:
        pd.DataFrame: DataFrame with centered and scaled 'value' column.
    """
    res = res.copy()
    res['scaling'] = np.power(res['n'], scale)

    def center(row):
        if row['metric'] == 'gap_FPR' and 'epsilon_pos' in row:
            return row['value'] - row['epsilon_pos']
        elif row['metric'] == 'gap_FNR' and 'epsilon_neg' in row:
            return row['value'] - row['epsilon_neg']
        else:
            return row['value'] - row[reference_col]

    res['value'] = res.apply(center, axis=1) * res['scaling']
    res = res.drop(columns='scaling')

    return res


def plot_metrics(df, row=None, col=None,
                 row_order=None, col_order=None,
                 reference_col=None, centered=False, **kwargs):
    """
    Plot metrics with confidence intervals and reference lines from simulation results.

    Args:
        df (pd.DataFrame): DataFrame with columns: 'n', 'metric', 'value', optionally
                           reference_col, 'epsilon_pos', 'epsilon_neg'.
        row (str or None): Variable to facet by row.
        col (str or None): Variable to facet by column.
        row_order (list): Order for rows.
        col_order (list): Order for columns.
        reference_col (str or None): Column to use for reference horizontal line.
        centered (bool): If True, draw horizontal reference lines at 0.
        **kwargs: Passed to sns.FacetGrid.

    Returns:
        sns.FacetGrid
    """
    # Treat 'n' as categorical
    n_arr = sorted(df['n'].unique())

    # Handle facet grid setup
    facet_kws = dict()
    if row:
        facet_kws['row'] = row
        if row_order is not None:
            facet_kws['row_order'] = row_order
    if col:
        facet_kws['col'] = col
        if col_order is not None:
            facet_kws['col_order'] = col_order

    g = sns.FacetGrid(df, margin_titles=True, despine=False, sharey=False, **facet_kws, **kwargs)
    g.map_dataframe(sns.pointplot, x='n', y='value', errorbar='sd', order=n_arr)

    for ax in g.axes.flat:
        ax.set_xticks(range(len(n_arr)))
        ax.set_xticklabels(n_arr, rotation=45)

    # Draw reference lines
    for key, ax in g.axes_dict.items():
        if isinstance(key, tuple):
            facet_row, facet_col = key if len(key) == 2 else (key[0], None)
        else:
            facet_row, facet_col = (key, None) if row else (None, key)

        m = facet_col or facet_row or df['metric'].unique()[0]
        facet_df = df.copy()
        if row and facet_row is not None:
            facet_df = facet_df[facet_df[row] == facet_row]
        if col and facet_col is not None:
            facet_df = facet_df[facet_df[col] == facet_col]

        if facet_df.empty:
            continue

        if centered:
            ax.axhline(0, ls='--', color='gray', alpha=0.7)
        else:
            if m == 'gap_FPR':
                if 'epsilon_pos' in facet_df.columns:
                    ref_val = facet_df['epsilon_pos'].iloc[0]
                elif reference_col in facet_df.columns:
                    ref_val = facet_df[reference_col].iloc[0]
                else:
                    ref_val = 0
                ax.axhline(ref_val, ls='--', color='gray', label='ε_pos')
            elif m == 'gap_FNR':
                if 'epsilon_neg' in facet_df.columns:
                    ref_val = facet_df['epsilon_neg'].iloc[0]
                elif reference_col in facet_df.columns:
                    ref_val = facet_df[reference_col].iloc[0]
                else:
                    ref_val = 0
                ax.axhline(ref_val, ls='--', color='gray', label='ε_neg')
            elif reference_col in facet_df.columns:
                ref_val = facet_df[reference_col].iloc[0]
                ax.axhline(ref_val, ls='--', color='gray', label='reference')

    g.set_titles(row_template='{row_name}', col_template='{col_name}')
    g.tight_layout()

    return g


def plot_noise(df, obj_true, pos_true, neg_true, errorbar='sd'):
    n_arr = sorted(df['n'].unique())
    xlim = (df['n'].min(), df['n'].max())

    df = df.melt(id_vars = ['n', 'id'])
    g = sns.FacetGrid(df, row = 'id', col = 'variable', xlim = xlim)
    g.map(sns.pointplot, 'n', 'value', order = n_arr, errorbar=errorbar)
    g.set_xticklabels(rotation = 45)

    for i, val in enumerate(obj_true):
        ax = g.axes[0, i]
        ax.hlines(val, *ax.get_xlim(), color = 'red')
    for i, val in enumerate(pos_true):
        ax = g.axes[1, i]
        ax.hlines(val, *ax.get_xlim(), color = 'red')
    for i, val in enumerate(neg_true):
        ax = g.axes[2, i]
        ax.hlines(val, *ax.get_xlim(), color = 'red')
    for i in range(3):
        ax = g.axes[i, 4]
        ax.hlines(0, *ax.get_xlim())
    
    g.figure.suptitle('Noisy Coefficients vs True Values, and L2 Distances', fontsize=16)
    g.tight_layout()
    
    return g
