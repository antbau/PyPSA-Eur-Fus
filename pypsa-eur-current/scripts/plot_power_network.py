# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Creates plots for optimised power network topologies and regional generation,
storage and conversion capacities built.
"""

import logging

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from _helpers import configure_logging, set_scenario_config
from plot_summary import preferred_order, rename_techs
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches

logger = logging.getLogger(__name__)


def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    if "heat pump" in tech or "resistive heater" in tech:
        return "power-to-heat"
    elif tech in ["H2 Electrolysis", "methanation", "H2 liquefaction"]:
        return "power-to-gas"
    elif tech == "H2":
        return "H2 storage"
    elif tech in ["NH3", "Haber-Bosch", "ammonia cracker", "ammonia store"]:
        return "ammonia"
    elif tech in ["OCGT", "CHP", "gas boiler", "H2 Fuel Cell"]:
        return "gas-to-power/heat"
    # elif "solar" in tech:
    #     return "solar"
    elif tech in ["Fischer-Tropsch", "methanolisation"]:
        return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif "CC" in tech or "sequestration" in tech:
        return "CCS"
    else:
        return tech


def assign_location(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -1:
                continue
            names = ifind.index[ifind == i]
            c.df.loc[names, "location"] = names.str[:i]


def load_projection(plotting_params):
    proj_kwargs = plotting_params.get("projection", dict(name="EqualEarth"))
    proj_func = getattr(ccrs, proj_kwargs.pop("name"))
    return proj_func(**proj_kwargs)


def plot_map(
    n,
    components=["links", "stores", "storage_units", "generators"],
    bus_size_factor=2e10,
    transmission=True,
    with_legend=True,
):
    tech_colors = snakemake.params.plotting["tech_colors"]

    assign_location(n)
    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    costs = pd.DataFrame(index=n.buses.index)

    for comp in components:
        df_c = getattr(n, comp)

        if df_c.empty:
            continue

        df_c["nice_group"] = df_c.carrier.map(rename_techs_tyndp)

        attr = "e_nom_opt" if comp == "stores" else "p_nom_opt"

        costs_c = (
            (df_c.capital_cost * df_c[attr])
            .groupby([df_c.location, df_c.nice_group])
            .sum()
            .unstack()
            .fillna(0.0)
        )
        costs = pd.concat([costs, costs_c], axis=1)

        logger.debug(f"{comp}, {costs}")

    costs = costs.T.groupby(costs.columns).sum().T

    costs.drop(list(costs.columns[(costs == 0.0).all()]), axis=1, inplace=True)

    new_columns = preferred_order.intersection(costs.columns).append(
        costs.columns.difference(preferred_order)
    )
    costs = costs[new_columns]

    for item in new_columns:
        if item not in tech_colors:
            logger.warning(f"{item} not in config/plotting/tech_colors")

    costs = costs.stack()  # .sort_index()

    # hack because impossible to drop buses...
    eu_location = snakemake.params.plotting.get("eu_node_location", dict(x=-5.5, y=46))
    n.buses.loc["EU gas", "x"] = eu_location["x"]
    n.buses.loc["EU gas", "y"] = eu_location["y"]

    n.links.drop(
        n.links.index[(n.links.carrier != "DC") & (n.links.carrier != "B2B")],
        inplace=True,
    )

    # drop non-bus
    to_drop = costs.index.levels[0].symmetric_difference(n.buses.index)
    if len(to_drop) != 0:
        logger.info(f"Dropping non-buses {to_drop.tolist()}")
        costs.drop(to_drop, level=0, inplace=True, axis=0, errors="ignore")

    # make sure they are removed from index
    costs.index = pd.MultiIndex.from_tuples(costs.index.values)

    # antbau: costs_transmit is a copy of costs, so that it can be used in plot_map_agg, as plot_map alters the n object
    costs_transmit = costs.copy()

    threshold = 100e6  # 100 mEUR/a
    carriers = costs.groupby(level=1).sum()
    carriers = carriers.where(carriers > threshold).dropna()
    carriers = list(carriers.index)

    # PDF has minimum width, so set these to zero
    line_lower_threshold = 500.0
    line_upper_threshold = 1e4
    linewidth_factor = 4e3
    ac_color = "rosybrown"
    dc_color = "darkseagreen"

    title = "added grid"

    if snakemake.wildcards["ll"] == "v1.0":
        # should be zero
        line_widths = n.lines.s_nom_opt - n.lines.s_nom
        link_widths = n.links.p_nom_opt - n.links.p_nom
        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            linewidth_factor = 2e3
            line_lower_threshold = 0.0
            title = "current grid"
    else:
        line_widths = n.lines.s_nom_opt - n.lines.s_nom_min
        link_widths = n.links.p_nom_opt - n.links.p_nom_min
        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            title = "total grid"

    line_widths = line_widths.clip(line_lower_threshold, line_upper_threshold)
    link_widths = link_widths.clip(line_lower_threshold, line_upper_threshold)

    line_widths = line_widths.replace(line_lower_threshold, 0)
    link_widths = link_widths.replace(line_lower_threshold, 0)

    fig, ax = plt.subplots(subplot_kw={"projection": proj})
    fig.set_size_inches(7, 6)

    n.plot(
        bus_sizes=costs / bus_size_factor,
        bus_colors=tech_colors,
        line_colors=ac_color,
        link_colors=dc_color,
        line_widths=line_widths / linewidth_factor,
        link_widths=link_widths / linewidth_factor,
        ax=ax,
        **map_opts,
    )

    sizes = [20, 10, 5]
    labels = [f"{s} bEUR/a" for s in sizes]
    sizes = [s / bus_size_factor * 1e9 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.01, 1.06),
        labelspacing=0.8,
        frameon=False,
        handletextpad=0,
        title="system cost",
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )

    sizes = [10, 5]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.27, 1.06),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
        title=title,
    )

    add_legend_lines(
        ax, sizes, labels, patch_kw=dict(color="lightgrey"), legend_kw=legend_kw
    )

    legend_kw = dict(
        bbox_to_anchor=(1.52, 1.04),
        frameon=False,
    )

    if with_legend:
        colors = [tech_colors[c] for c in carriers] + [ac_color, dc_color]
        labels = carriers + ["HVAC line", "HVDC link"]

        add_legend_patches(
            ax,
            colors,
            labels,
            legend_kw=legend_kw,
        )

    fig.savefig(snakemake.output.map, bbox_inches="tight")

    return costs_transmit

#antbau
# add function similar to plot_map but cluster into fossil, gas, nuclear, fusion, solar, wind, hydro
# similar to plot_map but uses costs_transmit from plot_map instead of recalculating the costs
def plot_map_agg(
    n,
    bus_size_factor=2e10,
    transmission=True,
    with_legend=True,
    costs_transmit=None,
):
    assign_location(n)

    # not a very nice solution, but plot_map is called before plot_map_agg and alters the n object
    costs = costs_transmit.copy()

    tech_groups = {
        "nuclear": ["nuclear", "uranium"],
        "fusion": ["fusion", "deu_tri"],
        "hydro": ["hydroelectricity"],
        "wind": ["onshore wind", "offshore wind"],
        "solar": ["solar PV", "solar rooftop", "solar thermal", "solar-hsat"],
        "other electricity incl power grid": ["V2G", "battery storage", "electricity distribution grid", "transmission lines", "BEV charger"],
        "gas": ["gas", "gas-to-power/heat", "power-to-heat", "biogas", "gas for industry", "power-to-gas", "solid biomass for industry"],
        "fossil": ["coal", "lignite", "oil", "oil primary", "coal for industry", "land transport oil", "oil refining", "shipping oil", "agriculture machinery oil"],
        "synfuel": ["industry methanol", "kerosene for aviation", "naptha for industry", "power-to-liquid", "shipping methanol", "methanol"],
        "H2": ["H2 pipeline", "SMR","H2 Store"],
        "heat": ["biomass boiler", "power-to-heat"],
        "CO2": ["DAC", "co2 sequestred", "process emissions"],
    }

    agg_tech_colors = {
        "fossil": "#000000",
        "gas": "#808080",
        "nuclear": "#A2AD00", 
        "fusion": "#0065BD", 
        "solar": "#E37222",
        "wind": "#98C6EA", 
        "hydro": "#64A0C8", 
        "CO2": "#E5DE77",
        "H2": "#BC8F8F",
        "other electricity incl power grid": "#DAD7CB",
        "heat": "#003359",
        "synfuel": "#8FBC8F",
    }

    # create tuples for the aggregated cost dataframe
    # all nodes from the costs dataframe and all technologies from the tech_groups dictionary
    index = pd.MultiIndex.from_tuples(
        [(node, tech) for node in costs.index.levels[0] for tech in tech_groups.keys()]
    )
    aggregated_costs = pd.DataFrame(index=index, columns=["value"])

    aggregated_costs.index = pd.MultiIndex.from_tuples(aggregated_costs.index.values)

    # Reverse the tech_groups dictionary to map individual technologies to their group
    tech_to_group = {tech: group for group, techs in tech_groups.items() for tech in techs}

    # Map the second-level index (technology) to their groups
    costs = costs.reset_index()  # Convert multi-index to columns
    costs = costs.rename(columns={costs.columns[-1]: "value"})  # Rename the last column to 'value'
    costs["tech_group"] = costs["level_1"].map(tech_to_group)

    # Drop rows where technology doesn't belong to any group
    costs = costs.dropna(subset=["tech_group"])

    # Aggregate costs by node and technology group
    aggregated_costs_calc = (
        costs.groupby(["level_0", "tech_group"])["value"]
        .sum()
        .reset_index()
        .rename(columns={"level_0": "node", "tech_group": "technology", "0": "value"})
    )

    # fill the aggregated_costs dataframe with the aggregated costs
    for node, technology, value in aggregated_costs_calc.values:
        aggregated_costs.loc[(node, technology)] = value

    # fill the aggregated_costs dataframe with zeros for the missing technologies
    for node in aggregated_costs.index.levels[0]:
        for technology in aggregated_costs.index.levels[1]:
            if aggregated_costs.loc[(node, technology)].isnull().values[0]:
                aggregated_costs.loc[(node, technology)] = 0

    # antbau: if tutorial config, set threshold to 1e4, else 100e6
    if snakemake.config["tutorial"]:
        threshold = 0 # 10 kEUR/a
    else:
        threshold = 100e6  # 100 mEUR/a
    
    aggregated_carriers = aggregated_costs.groupby(level=1).sum()
    aggregated_carriers = aggregated_carriers.where(aggregated_carriers > threshold).dropna()
    aggregated_carriers = list(aggregated_carriers.index)

    # sort aggregated_carriers by the order in tech_groups
    aggregated_carriers = [tech for tech in tech_groups.keys() if tech in aggregated_carriers]

    # PDF has minimum width, so set these to zero
    line_lower_threshold = 500.0
    line_upper_threshold = 1e4
    linewidth_factor = 4e3
    ac_color = "#DAD7CB"
    dc_color = "#DAD7CB"

    title = "added grid"

    if snakemake.wildcards["ll"] == "v1.0":
        # should be zero
        line_widths = n.lines.s_nom_opt - n.lines.s_nom
        link_widths = n.links.p_nom_opt - n.links.p_nom
        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            linewidth_factor = 2e3
            line_lower_threshold = 0.0
            title = "current grid"
    else:
        line_widths = n.lines.s_nom_opt - n.lines.s_nom_min
        link_widths = n.links.p_nom_opt - n.links.p_nom_min
        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            title = "total grid"

    line_widths = line_widths.clip(line_lower_threshold, line_upper_threshold)
    link_widths = link_widths.clip(line_lower_threshold, line_upper_threshold)

    line_widths = line_widths.replace(line_lower_threshold, 0)
    link_widths = link_widths.replace(line_lower_threshold, 0)

    fig, ax = plt.subplots(subplot_kw={"projection": proj})
    fig.set_size_inches(7, 6)

    aggregated_costs_series = aggregated_costs["value"]

    # Make sure the index aligns with n.buses.index
    missing_buses = set(n.buses.index) - set(aggregated_costs_series.index.get_level_values(0))
    if missing_buses:
        for bus in missing_buses:
            for tech in tech_groups.keys():
                aggregated_costs_series.loc[(bus, tech)] = 0

    # Ensure the Series index is sorted to avoid alignment issues
    aggregated_costs_series = aggregated_costs_series.sort_index()

    n.plot(
        bus_sizes=aggregated_costs_series / bus_size_factor,
        bus_colors=agg_tech_colors,
        line_colors=ac_color,
        link_colors=dc_color,
        line_widths=line_widths / linewidth_factor,
        link_widths=link_widths / linewidth_factor,
        ax=ax,
        **map_opts,
    )

    sizes = [20, 10, 5]
    labels = [f"{s} bEUR/a" for s in sizes]
    sizes = [s / bus_size_factor * 1e9 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.01, 1.06),
        labelspacing=0.8,
        frameon=False,
        handletextpad=0,
        title="system cost",
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )

    sizes = [10, 5]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.27, 1.06),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
        title=title,
    )

    add_legend_lines(
        ax, sizes, labels, patch_kw=dict(color="lightgrey"), legend_kw=legend_kw
    )

    legend_kw = dict(
        bbox_to_anchor=(1.52, 1.04),
        frameon=False,
    )

    if with_legend:
        colors = [agg_tech_colors[c] for c in aggregated_carriers]
        labels = aggregated_carriers
        add_legend_patches(
            ax,
            colors,
            labels,
            legend_kw=legend_kw,
        )

    fig.savefig(snakemake.output.aggregated_map, bbox_inches="tight")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_power_network",
            opts="",
            clusters="37",
            ll="v1.0",
            sector_opts="4380H-T-H-B-I-A-dist1",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)

    regions = gpd.read_file(snakemake.input.regions).set_index("name")

    map_opts = snakemake.params.plotting["map"]

    if map_opts["boundaries"] is None:
        map_opts["boundaries"] = regions.total_bounds[[0, 2, 1, 3]] + [-1, 1, -1, 1]

    proj = load_projection(snakemake.params.plotting)

    cost_transmit = plot_map(n)

    # antbau: add plot_map_agg function
    plot_map_agg(n, costs_transmit=cost_transmit)