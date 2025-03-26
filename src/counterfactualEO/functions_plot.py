from collections.abc import Iterable
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


def combine_coefs(results, n_arr):
    """Combine LP coefficients from simulation output into a dataframe."""
    obj = pd.DataFrame(
        {'component': 'L2', 'n': n_arr, 'value': res['dist_obj'], 'id': 'obj'})

    coefs_noisy = pd.concat([
        to_df(results['obj_noisy'], n_arr, 'obj'),
        to_df(results['pos_noisy'], n_arr, 'pos'),
        to_df(results['neg_noisy'], n_arr, 'neg')
    ])


def plot_noise(df, obj_true, pos_true, neg_true, ci='sd'):
    xlim = (df['n'].min(), df['n'].max())

    df = df.melt(id_vars = ['n', 'id'])
    g = sns.FacetGrid(df, row = 'id', col = 'variable', xlim = xlim)
    g.map(sns.pointplot, 'n', 'value', order = n_arr, ci = ci)
    g.set_xticklabels(rotation = 45)

    for i, val in enumerate(obj_true):
        ax = g.axes[0, i]
        ax.hlines(val, *ax.get_xlim(), color = 'red')
    for i, val in enumerate(pos_true):
        ax = g.axes[1, i]
        ax.hlines(0, *ax.get_xlim())
        ax.hlines(val, *ax.get_xlim(), color = 'red')
    for i, val in enumerate(neg_true):
        ax = g.axes[2, i]
        ax.hlines(0, *ax.get_xlim())
        ax.hlines(val, *ax.get_xlim(), color = 'red')
    for i in range(3):
        ax = g.axes[i, 4]
        ax.hlines(0, *ax.get_xlim())


def plot_coefs(results):
    """Plot 'true' and noisy coefficients."""
    coefs_noisy = pd.concat([
        arr_to_df(results['obj_noisy'], n_arr, 'obj'),
        vec_to_df(results['dist_obj'], n_arr, 'obj'),
        arr_to_df(results['pos_noisy'], n_arr, 'pos'),
        vec_to_df(results['dist_pos'], n_arr, 'pos'),
        arr_to_df(results['neg_noisy'], n_arr, 'neg'),
        vec_to_df(results['dist_neg'], n_arr, 'neg')
    ])

    xlim = (min(n_arr), max(n_arr))
    ylim = (-1.1, 1.1)

    g = sns.FacetGrid(coefs_noisy, row = 'id', col = 'component', xlim = xlim,
                     ylim = ylim)
    g.map(sns.pointplot, 'n', 'value', order = n_arr)
    g.set_xticklabels(rotation = 45)

    for i, val in enumerate(results['obj_true']):
        ax = g.axes[0, i]
        ax.hlines(val, *ax.get_xlim())
    for i, val in enumerate(results['pos_true']):
        ax = g.axes[1, i]
        ax.hlines(0, *ax.get_xlim(), linestyle = '--', color = 'red')
        ax.hlines(val, *ax.get_xlim())
    for i, val in enumerate(results['neg_true']):
        ax = g.axes[2, i]
        ax.hlines(0, *ax.get_xlim(), linestyle = '--', color = 'red')
        ax.hlines(val, *ax.get_xlim())


def plot_metrics(results, epsilon_pos, epsilon_neg):
    """Plot risk and fairness gaps."""
    ## Plot risk and fairness gaps as a function of sample size,
    ## with true minimum risk and true fairness gaps for reference.

    metrics_Y0 = pd.concat(results['metrics_Y0_noisy'], keys=n_arr)
    metrics_Y0 = metrics_Y0.reset_index().drop(columns='level_1').rename(
        columns={'level_0': 'n'})
    metrics_Y = pd.concat(results['metrics_Y_noisy'], keys=n_arr)
    metrics_Y = metrics_Y.reset_index().drop(columns='level_1').rename(
        columns={'level_0': 'n'})
    metrics = pd.concat([metrics_Y0, metrics_Y])

    m = results['metrics_Y0_best']
    risk = m.loc[m.Metric == 'Risk', 'Value'].values[0]

    g = sns.FacetGrid(metrics, row='Outcome', col='Metric',
                      col_order=['risk', 'gap_FPR', 'gap_FNR'])
    g.map(sns.pointplot, 'n', 'value', order=n_arr)
    g.set_xticklabels(rotation=45)

    g.axes[0, 0].hlines(risk, *g.axes[0, 0].get_xlim())
    g.axes[1, 0].hlines(risk, *g.axes[1, 0].get_xlim())

    g.axes[0, 1].hlines(epsilon_pos, *g.axes[0, 1].get_xlim())
    g.axes[1, 1].hlines(epsilon_pos, *g.axes[1, 1].get_xlim())

    g.axes[0, 2].hlines(epsilon_neg, *g.axes[0, 2].get_xlim())
    g.axes[1, 2].hlines(epsilon_neg, *g.axes[1, 2].get_xlim())


def to_iterable(obj):
    if not isinstance(obj, Iterable):
        obj = [obj]
    return obj


def plot_metrics2(df, n_arr, risk_best, epsilon_pos, epsilon_neg, row, col,
                  **kwargs):
    """Plot metrics for Task (1) simulations.

    Need to be able to accommodate either one or multiple settings of the epsilons.
    """
    xlim = (min(n_arr), max(n_arr))
    #     g = sns.FacetGrid(df, row = row, col = col,
    #                       col_order = ['risk', 'gap_FPR', 'gap_FNR'], xlim = xlim,
    #                      ylim = (0, 1), **kwargs)
    g = sns.FacetGrid(df, row=row, col=col,
                      col_order=['risk', 'gap_FPR', 'gap_FNR'], xlim=xlim,
                      **kwargs)
    g.map(sns.pointplot, 'n', 'value', order=n_arr, ci='sd')
    g.set_xticklabels(rotation=45)

    risk_best = to_iterable(risk_best)
    epsilon_pos = to_iterable(epsilon_pos)
    epsilon_neg = to_iterable(epsilon_neg)

    for i, rr in enumerate(risk_best):
        g.axes[i, 0].hlines(rr, *g.axes[i, 0].get_xlim())
        g.axes[i, 0].hlines(rr, *g.axes[i, 0].get_xlim())

    for i, ee in enumerate(epsilon_pos):
        g.axes[i, 1].hlines(ee, *g.axes[i, 1].get_xlim())
        g.axes[i, 1].hlines(ee, *g.axes[i, 1].get_xlim())

    for i, ee in enumerate(epsilon_neg):
        g.axes[i, 2].hlines(ee, *g.axes[i, 1].get_xlim())
        g.axes[i, 2].hlines(ee, *g.axes[i, 1].get_xlim())

    g.set_titles(template='')

    for ax, m in zip(g.axes[0, :], ['risk', 'gap_FPR', 'gap_FNR']):
        ax.set_title(m)
    for ax, l in zip(g.axes[:, 0], df[row].unique()):
        ax.set_ylabel(l, rotation=90, ha='center', va='center')

    return g


def plot_metrics3(df, n_arr, risk_best, epsilon_pos, epsilon_neg, row, col,
                  **kwargs):
    """Plot metrics for Task (1) simulations.

    Need to be able to accommodate either one or multiple settings of the epsilons.
    """
    xlim = (min(n_arr), max(n_arr))
    #     g = sns.FacetGrid(df, row = row, col = col,
    #                       col_order = ['risk', 'gap_FPR', 'gap_FNR'], xlim = xlim,
    #                      ylim = (0, 1), **kwargs)
    g = sns.FacetGrid(df, row=row, col=col,
                      row_order=['risk', 'gap_FPR', 'gap_FNR'], xlim=xlim,
                      **kwargs)
    g.map(sns.pointplot, 'n', 'value', order=n_arr, ci='sd')
    g.set_xticklabels(rotation=45)

    risk_best = to_iterable(risk_best)
    epsilon_pos = to_iterable(epsilon_pos)
    epsilon_neg = to_iterable(epsilon_neg)

    for i, rr in enumerate(risk_best):
        g.axes[0, i].hlines(rr, *g.axes[0, i].get_xlim())
        g.axes[0, i].hlines(rr, *g.axes[0, i].get_xlim())

    for i, ee in enumerate(epsilon_pos):
        g.axes[1, i].hlines(ee, *g.axes[1, i].get_xlim())
        g.axes[1, i].hlines(ee, *g.axes[1, i].get_xlim())

    for i, ee in enumerate(epsilon_neg):
        g.axes[2, i].hlines(ee, *g.axes[2, i].get_xlim())
        g.axes[2, i].hlines(ee, *g.axes[2, i].get_xlim())

    g.set_titles(template='')
    for ax, m in zip(g.axes[0, :], df[col].unique()):
        ax.set_title(m)

    return g


def transform_metrics(res, risk, risk_change, epsilon_pos, epsilon_neg,
                      scale=0.5,
                      id_vars=['mc_iter', 'n']):
    """'Center' and scale the raw values.

    Args:
      res: the DataFrame of metrics.
      risk: iterable of reference risk values.
      epsilon_pos: iterable of reference epsilon_pos values.
      epsilon_neg: iterable of reference epsilon_neg values.
    """
    settings = res.scenario.unique()
    out = res.pivot_table(index=id_vars, columns=['metric', 'scenario'],
                          values='value')

    mult = np.power(out.index.get_level_values('n'), scale)
    if risk:
        for ss, rr in zip(settings, risk):
            out.loc[:, ('risk', ss)] = mult * (out.loc[:, ('risk', ss)] - rr)
    if risk_change:
        for ss, rr in zip(settings, risk_change):
            out.loc[:, ('risk_change', ss)] = mult * (
                        out.loc[:, ('risk_change', ss)] - rr)
    if epsilon_pos:
        for ss, pp in zip(settings, epsilon_pos):
            out.loc[:, ('gap_FPR', ss)] = mult * (
                (out.loc[:, ('gap_FPR', ss)] - pp))
    if epsilon_neg:
        for ss, nn in zip(settings, epsilon_neg):
            out.loc[:, ('gap_FNR', ss)] = mult * (
                (out.loc[:, ('gap_FNR', ss)] - nn))

    out = out.reset_index().melt(id_vars=id_vars)

    return out


def plot_metrics_est(df, metrics_pre, metrics_post, n_arr, row='scenario',
                     col='metric', **kwargs):
    """Plot metrics for Task (2) simulations.

    Need to be able to accommodate either one or multiple settings of the epsilons.
    """
    risk_pre = metrics_pre.query("metric=='risk'")['value'].values[0]
    risk_post = metrics_post.query("metric=='risk'")['value'].values[0]
    risk_change = metrics_post.query("metric=='risk_change'")['value'].values[0]
    gap_FPR = metrics_post.query("metric=='gap_FPR'")['value'].values[0]
    gap_FNR = metrics_post.query("metric == 'gap_FNR'")['value'].values[0]

    xlim = (min(n_arr), max(n_arr))
    g = sns.FacetGrid(df, row=row, col=col,
                      col_order=['risk', 'risk_change', 'gap_FPR', 'gap_FNR'],
                      xlim=xlim, **kwargs)
    g.map(sns.pointplot, 'n', 'value', order=n_arr, ci='sd')
    g.set_xticklabels(rotation=45)

    g.set_titles(template='')

    for ax, m in zip(g.axes[0, :],
                     ['risk', 'risk_change', 'gap_FPR', 'gap_FNR']):
        ax.set_title(m)
    for ax, l in zip(g.axes[:, 0], df[row].unique()):
        ax.set_ylabel(l, rotation=90, ha='center', va='center')

    for i in range(g.axes.shape[0]):
        g.axes[i, 0].hlines(risk_post, *g.axes[i, 0].get_xlim())
        g.axes[i, 1].hlines(risk_change, *g.axes[i, 1].get_xlim())
        g.axes[i, 2].hlines(gap_FPR, *g.axes[i, 2].get_xlim())
        g.axes[i, 3].hlines(gap_FNR, *g.axes[i, 3].get_xlim())
