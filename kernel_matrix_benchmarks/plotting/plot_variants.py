# Choices of (x_axis, y_axis) pairs that we want to display
# on the website in the "details" page.
# The cost and performance keys are defined in all_metrics in metrics.py.
all_plot_variants = {
    cost + "/" + perf: (cost, perf)
    for cost in ["total-time", "query-time", "memory-footprint"]
    for perf in ["rmse-error", "max-error", "mean-error"]
}
