"""
this module contains codes that are responsible to do all the visualizations,
from plots to drawings and even gephy input file generation
"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns

import pyx
from pyx import canvas, path, style, deco, color
# from pyx import *

import utils
import constants as cs

sns.set(color_codes=True)


# seaborn functions
def histogram(data, file_name="test.svg", file_path=cs.SVG_PATH):
    sns.distplot(data, rug=False, kde=False, hist=True,
                 bins=8, fit=sp.stats.powerlaw)
    svg_file = utils.join_path(file_path, file_name)
    sns.plt.savefig(svg_file)


def cluster_vertices(bio_net, file_name, file_path=cs.SVG_PATH):
    plt.figure()

    plt.subplot(211)
    ax = plt.gca()
    sns.distplot(bio_net.org_cluster.cl_size1, rug=False,
                 ax=ax, kde=False, fit=sp.stats.powerlaw)

    plt.subplot(212)
    ax = plt.gca()
    sns.distplot(bio_net.org_cluster.cl_size2, rug=False,
                 ax=ax, kde=False, fit=sp.stats.powerlaw)

    svg_file = utils.join_path(file_path, file_name)
    plt.savefig(svg_file)


@utils.time_it
def visualise_org_degree(organism, file_path=cs.SVG_PATH):
    file_name = '{}-degree.svg'.format(organism.org_id)

    plt.figure()

    ax = plt.gca()
    sns.distplot(organism.degree, rug=False, ax=ax,
                 kde=False, bins=50, fit=sp.stats.powerlaw)

    svg_file = utils.join_path(file_path, file_name)
    plt.savefig(svg_file)


@utils.time_it
def sim_degree(bio_net, file_path=cs.SVG_PATH):
    file_name = '{}-{}-sim<{}>-degree.svg'.format(
        bio_net.org1.org_id, bio_net.org2.org_id, bio_net.similarity_mode)

    data = {}
    data["degree geometric average"] = []
    data["normal sim score"] = []

    for i in range(bio_net.org1.node_count):
        for j in range(bio_net.org2.node_count):
            if (bio_net.similarity[bio_net.v_ind(i, j)] > cs.MIN_VIS_CUT):
                data["degree geometric average"].append(
                    (bio_net.org1.degree[i] * bio_net.org2.degree[j])**0.5)
                data["normal sim score"].append(
                    bio_net.similarity[bio_net.v_ind(i, j)])

    df = pd.DataFrame(data)

    im = sns.lmplot(x="degree geometric average", y="normal sim score",
                    data=df, scatter_kws={"s": 5}, fit_reg=False)
    mx = max(bio_net.similarity)
    im.set(ylim=((-mx * cs.NORM_MARGIN), (mx * (1 + cs.NORM_MARGIN))))

    svg_file = utils.join_path(file_path, file_name)
    sns.plt.savefig(svg_file)


@utils.time_it
def sim_degree_3d(bio_net, file_path=cs.SVG_PATH):
    file_name = '{}-{}-sim<{}>-degree-3d.svg'.format(
        bio_net.org1.org_id, bio_net.org2.org_id, bio_net.similarity_mode)

    data = {}
    data["degree of {}".format(bio_net.org1.org_id)] = []
    data["degree of {}".format(bio_net.org2.org_id)] = []
    data["normal sim score"] = []

    for i in range(bio_net.org1.node_count):
        for j in range(bio_net.org2.node_count):
            if (bio_net.similarity[bio_net.v_ind(i, j)] > cs.MIN_VIS_CUT):
                data["degree of {}".format(bio_net.org1.org_id)].append(
                    bio_net.org1.degree[i])
                data["degree of {}".format(bio_net.org2.org_id)].append(
                    bio_net.org2.degree[j])
                data["normal sim score"].append(
                    bio_net.similarity[bio_net.v_ind(i, j)])

    x1 = data["degree of {}".format(bio_net.org1.org_id)]
    x2 = data["degree of {}".format(bio_net.org2.org_id)]
    z = data["normal sim score"]

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x1, x2, z, c=z, cmap=cm.jet)

    ax.set_xlabel("degree of {}".format(bio_net.org1.org_id))
    ax.set_ylabel("degree of {}".format(bio_net.org2.org_id))
    ax.set_zlabel('similarity score')

    svg_file = utils.join_path(file_path, file_name)
    plt.savefig(svg_file)


# pyx functions
def test_pie(radius, start, end):
    c = canvas.canvas()
    container = path.rect(-(radius + 1), -(radius + 1),
                          2 * (radius + 1), 2 * (radius + 1))

    c.stroke(container, [style.linewidth(0.001), color.rgb.red])

    pie = path.path(path.moveto(0, 0), path.arc(
        0, 0, radius, start, end), path.closepath())

    c.stroke(pie, [style.linewidth(0.1), pyx.color.rgb(1, 1, 1),
                   deco.filled([color.rgb.red])])
    c.writeSVGfile("figure")


def draw_pie(c, radius, start, end):
    pie = path.path(path.moveto(0, 0), path.arc(
        0, 0, radius, start, end), path.closepath())

    hue = (start + end) / (360 * 2)
    color = pyx.color.hsb(hue, 0.8, 0.8)

    c.stroke(pie, [style.linewidth(0.01), pyx.color.rgb(1, 1, 1),
                   deco.filled([color])])


def recurse_sunburst(c, graph, level, start, end):
    size = graph['size']
    children = graph['clusters']
    current_point = start
    for child in children:
        cluster = children[child]
        width = (cluster['size'] / size) * (end - start)
        recurse_sunburst(c, cluster, level + 1, current_point,
                         current_point + width)
        current_point += width
    draw_pie(c, level + 1, start, end)


def cluster_sunburst(organism, graph, method):
    c = canvas.canvas()

    core_length = 5
    recurse_sunburst(c, graph, core_length, 0, 360)
    core = path.circle(0, 0, core_length)
    c.stroke(core, [style.linewidth(0.1), pyx.color.rgb(1, 1, 1),
                    deco.filled([pyx.color.rgb(1, 1, 1)])])

    file_name = '{}-{}-clusters_sunburst.svg'.format(organism.org_id, method)
    with open(utils.join_path(cs.SVG_PATH, file_name), 'wb') as svg:
        c.writeSVGfile(svg)


# gephi functions
def gephi_organism_ppi(organism, file_path=cs.GEPHI_PATH):
    file_name = '{}-organism.gdf'.format(organism.org_id)
    with open(utils.join_path(file_path, file_name), 'w') as gdf:
        gdf.write('nodedef>name VARCHAR,label VARCHAR\n')
        for nid in organism.id_to_node:
            gdf.write('n{},{}\n'.format(nid, organism.id_to_node[nid]))
        gdf.write('edgedef>node1 VARCHAR,node2 VARCHAR,directed BOOLEAN\n')
        for edge in organism.edges:
            gdf.write('n{},n{},false\n'.format(edge[0], edge[1]))


def gephi_network_aligned(alignment, bio_net, file_path=cs.GEPHI_PATH):
    file_name = '{}_{}-{}{}_alignment_{}.gdf'.format(
        bio_net.org1.org_id, bio_net.org2.org_id,
        bio_net.similarity_mode,
        bio_net.status, alignment.method)
    with open(utils.join_path(file_path, file_name), 'w') as gdf:
        gdf.write('nodedef>name VARCHAR,label VARCHAR\n')
        for pair in alignment.pairs:
            gdf.write(
                'a{}b{},{}/{}\n'.format(pair[0],
                                        pair[1],
                                        bio_net.org1.id_to_node[pair[0]],
                                        bio_net.org2.id_to_node[pair[1]],))
        gdf.write('edgedef>node1 VARCHAR,node2 VARCHAR,directed BOOLEAN\n')
        for edge in alignment.pair_edges:
            gdf.write('a{}b{},a{}b{},false\n'.format(
                edge[0][0], edge[0][1], edge[1][0], edge[1][1]))


def gephi_network_aligned_comp(alignment, bio_net, file_path=cs.GEPHI_PATH):
    file_name = '{}_{}-{}{}_comparative_alignment_{}.gdf'.format(
        bio_net.org1.org_id, bio_net.org2.org_id,
        bio_net.similarity_mode,
        bio_net.status, alignment.method)
    with open(utils.join_path(file_path, file_name), 'w') as gdf:
        gdf.write('nodedef>name VARCHAR,label VARCHAR,level INT\n')
        for pair in alignment.pairs:
            gdf.write(
                'a{}b{},{}/{},1\n'.format(pair[0],
                                          pair[1],
                                          bio_net.org1.id_to_node[pair[0]],
                                          bio_net.org2.id_to_node[pair[1]],))
        for nid in bio_net.org1.id_to_node:
            gdf.write('a{},{},0\n'.format(nid,
                                          bio_net.org1.id_to_node[nid]))
        for nid in bio_net.org2.id_to_node:
            gdf.write('b{},{},2\n'.format(nid,
                                          bio_net.org2.id_to_node[nid]))
        gdf.write('edgedef>node1 VARCHAR,node2 VARCHAR,'
                  'directed BOOLEAN,weight DOUBLE,visible BOOLEAN\n')
        for edge in alignment.pair_edges:
            gdf.write('a{}b{},a{}b{},false,1.0,true\n'.format(
                edge[0][0], edge[0][1], edge[1][0], edge[1][1]))
        for edge in bio_net.org1.edges:
            gdf.write('a{},a{},false,1.0,true\n'.format(edge[0], edge[1]))
        for edge in bio_net.org2.edges:
            gdf.write('b{},b{},false,1.0,true\n'.format(edge[0], edge[1]))
        for pair in alignment.pairs:
            gdf.write('a{}b{},a{},false,1000.0,false\n'.format(pair[0],
                                                               pair[1],
                                                               pair[0]))
            gdf.write('a{}b{},b{},false,1000.0,false\n'.format(pair[0],
                                                               pair[1],
                                                               pair[1]))
