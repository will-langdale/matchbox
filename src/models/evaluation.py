# TODO: model metrics

# file_name = "m_u_parameters_chart_" + timestr + ".html"
# logging.info("Saving m and u probabilities chart to %s", file_name)
# c = linker.m_u_parameters_chart()
# output_file = Path(loc.OUTPUTS_HOME, "figures", file_name)
# save_offline_chart(c.spec, filename=output_file, overwrite=True)

# file_name = "cluster_studio_" + timestr + ".html"
# logging.info("Saving cluster studio to %s", file_name)
# output_file = Path(loc.OUTPUTS_HOME, "figures", file_name)
# linker.cluster_studio_dashboard(predictions, clusters, out_path=output_file)

# look at how many values are missing in the various columns
# timestr = time.strftime("%Y%m%d-%H%M%S")
# c = linker.completeness_chart()
# file_name = "completeness_chart_" + timestr + ".html"
# logging.info("Saving completeness chart to %s", file_name)
# output_file = Path(loc.OUTPUTS_HOME, "figures", file_name)
# save_offline_chart(c.spec, filename=output_file, overwrite=True)

# c = linker.missingness_chart()
# file_name = "missingness_chart_" + timestr + ".html"
# logging.info("Saving missingness chart to %s", file_name)
# output_file = Path(loc.OUTPUTS_HOME, "figures", file_name)
# save_offline_chart(c.spec, filename=output_file, overwrite=True)

# file_name = "comparison_viewer_" + timestr + ".html"
# logging.info("Saving comparison viewer to %s", file_name)
# output_file = Path(loc.OUTPUTS_HOME, "figures", file_name)
# linker.comparison_viewer_dashboard(predictions, output_file, overwrite=True)
